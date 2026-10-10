"""Image input handling: PIL image, path, URL, raw bytes or file-like object."""

from __future__ import annotations

import io
import os
import urllib.request
from typing import IO, TYPE_CHECKING, Union

from .exceptions import ImageLoadError

if TYPE_CHECKING:  # pragma: no cover
    from PIL import Image

ImageInput = Union["Image.Image", str, os.PathLike, bytes, bytearray, IO[bytes]]

DEFAULT_MAX_BYTES = 25 * 1024 * 1024


def _open_bytes(data: bytes):
    from PIL import Image

    with Image.open(io.BytesIO(data)) as im:
        return im.convert("RGB")


def _download(url: str, timeout: float, max_bytes: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "TestAtlasVision/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ImageLoadError(f"Image at {url} is larger than {max_bytes} bytes")
    return data


def load_image(
    src: ImageInput,
    *,
    allow_remote: bool = True,
    timeout: float = 30.0,
    max_bytes: int = DEFAULT_MAX_BYTES,
):
    """Return an RGB PIL image.

    Set `allow_remote=False` in any service that accepts untrusted input: fetching
    arbitrary URLs from a server is an SSRF risk. Accept uploads/bytes there instead.
    """
    from PIL import Image

    try:
        if isinstance(src, Image.Image):
            return src.convert("RGB")
        if isinstance(src, (bytes, bytearray)):
            return _open_bytes(bytes(src))
        if hasattr(src, "read"):
            return _open_bytes(src.read())
        if isinstance(src, (str, os.PathLike)):
            s = os.fspath(src)
            if s.startswith(("http://", "https://")):
                if not allow_remote:
                    raise ImageLoadError("Remote image URLs are disabled for this client")
                return _open_bytes(_download(s, timeout, max_bytes))
            with Image.open(s) as im:
                return im.convert("RGB")
    except ImageLoadError:
        raise
    except Exception as e:  # noqa: BLE001 - normalise everything PIL/urllib can throw
        raise ImageLoadError(f"Could not load image: {e}") from e
    raise ImageLoadError(f"Unsupported image input type: {type(src).__name__}")
