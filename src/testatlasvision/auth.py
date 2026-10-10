"""Hugging Face authentication for the gated N-ATLaS model.

Token lookup order (first hit wins):
  1. `token=` passed to the client
  2. env var HF_TOKEN, then HUGGING_FACE_HUB_TOKEN
  3. Colab Secret named HF_TOKEN (Colab sidebar -> key icon)
  4. a token cached by `huggingface-cli login` / `notebook_login()`
  5. if `interactive=True`, a hidden prompt

The token is never logged or stored by the SDK.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .exceptions import AuthenticationError, GatedModelAccessError, ModelLoadError

logger = logging.getLogger("testatlasvision")

ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def mask_token(token: Optional[str]) -> str:
    if not token:
        return "<none>"
    return token[:4] + "..." + token[-2:] if len(token) > 8 else "***"


def _colab_secret() -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore

        return userdata.get("HF_TOKEN") or None
    except Exception:  # not in Colab, secret missing, or notebook access not granted
        return None


def _cached_token() -> Optional[str]:
    try:
        from huggingface_hub import get_token

        return get_token() or None
    except Exception:
        return None


def resolve_token(token: Optional[str] = None, *, interactive: bool = False) -> Optional[str]:
    """Find a Hugging Face token, or return None if there isn't one."""
    if token is not None:
        token = token.strip()
        if not token:
            raise AuthenticationError("An empty Hugging Face token was passed.")
        return token

    for var in ENV_VARS:
        value = os.environ.get(var, "").strip()
        if value:
            logger.debug("Using Hugging Face token from %s", var)
            return value

    for finder, label in ((_colab_secret, "Colab secret"), (_cached_token, "cached login")):
        value = finder()
        if value:
            logger.debug("Using Hugging Face token from %s", label)
            return value

    if interactive:
        from getpass import getpass

        value = getpass("Hugging Face token (hidden): ").strip()
        return value or None
    return None


_HOW_TO_FIX = (
    "Create a token at https://huggingface.co/settings/tokens (read access is enough), "
    "request access on the model page, then pass it as TestAtlasVision(token=...), set the "
    "HF_TOKEN env var, or add a Colab Secret named HF_TOKEN."
)


def _probe(llm: str, token: Optional[str]) -> None:
    """Ask the Hub (not the local cache) whether `token` can read `llm`.

    NOTE: hf_hub_download() is not a valid access check on its own: when the server call
    fails (even with a 401/403) it silently returns a previously cached file instead.
    """
    import huggingface_hub as hub

    check = getattr(hub, "auth_check", None)
    if check is not None:
        check(llm, token=token)
    else:  # older huggingface_hub: force a real server round trip
        hub.hf_hub_download(llm, "config.json", token=token, force_download=True)


def _status(exc) -> Optional[int]:
    return getattr(getattr(exc, "response", None), "status_code", None)


def ensure_llm_access(llm: str, token: Optional[str]) -> None:
    """Fail fast, with a useful message, if we can't read the (gated) base model.

    One small API call; skipped for local paths. If the Hub can't be reached at all (offline) the
    check is skipped with a warning so cached/offline deployments still start.
    """
    if os.path.isdir(llm):
        return

    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

    try:
        _probe(llm, token)
    except GatedRepoError as e:  # must precede RepositoryNotFoundError (it is a subclass)
        if token is None or _status(e) == 401:
            raise AuthenticationError(
                f"{llm} is gated and no valid Hugging Face token was found (missing, invalid or expired). {_HOW_TO_FIX}"
            ) from e
        raise GatedModelAccessError(
            f"Your token is valid but its account has not been granted access to {llm}. "
            f"Open https://huggingface.co/{llm} while logged in and request access."
        ) from e
    except RepositoryNotFoundError as e:
        if token is None or _status(e) == 401:
            raise AuthenticationError(
                f"Could not access {llm}: the token is missing, invalid or expired (or the repo is private/gated). {_HOW_TO_FIX}"
            ) from e
        raise ModelLoadError(f"Repository {llm} was not found (check the name and token permissions).") from e
    except HfHubHTTPError as e:
        status = _status(e)
        if status in (401, 403):
            raise AuthenticationError(
                f"Hugging Face rejected the token for {llm} (HTTP {status}). {_HOW_TO_FIX}"
            ) from e
        raise ModelLoadError(f"Hugging Face error while checking {llm}: {e}") from e
    except Exception as e:  # noqa: BLE001 - network down etc.; from_pretrained will use the cache or fail clearly
        logger.warning("Could not verify access to %s (%s: %s); continuing.", llm, type(e).__name__, e)
