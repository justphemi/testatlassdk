"""CPU-only tests. They never load a model; GPU behaviour is tested via colab/smoke_test.ipynb."""

import io

import pytest
from PIL import Image

from testatlasvision import (
    AuthenticationError,
    GenerationConfig,
    ImageLoadError,
    Session,
    TestAtlasVision,
    UnsupportedLanguageError,
    load_image,
    normalize_language,
)
from testatlasvision.auth import mask_token, resolve_token
from testatlasvision.config import ASSIST_HEADER, EOT, USER_HEADER
from testatlasvision.prompting import build_conversation_text, build_text_prompt
from testatlasvision.types import Turn

# ---- languages --------------------------------------------------------------


@pytest.mark.parametrize(
    "value,code", [("ig", "ig"), ("IG", "ig"), ("Igbo", "ig"), (" yoruba ", "yo"), ("Hausa", "ha"), ("English", "en")]
)
def test_normalize_language(value, code):
    assert normalize_language(value) == code


def test_unsupported_language():
    with pytest.raises(UnsupportedLanguageError):
        normalize_language("fr")


# ---- prompts ----------------------------------------------------------------


def test_first_turn_has_no_extra_user_header():
    assert build_conversation_text([], "Hi?") == "Hi?" + ASSIST_HEADER


def test_follow_up_turns_get_user_header():
    hist = [Turn("Q1", "A1")]
    text = build_conversation_text(hist, "Q2")
    assert text == "Q1" + ASSIST_HEADER + "A1" + EOT + USER_HEADER + "Q2" + ASSIST_HEADER


def test_three_turns():
    hist = [Turn("Q1", "A1"), Turn("Q2", "A2")]
    text = build_conversation_text(hist, "Q3")
    assert text.count(USER_HEADER) == 2  # turns 2 and 3 only; turn 1's header is the prefix
    assert text.endswith("Q3" + ASSIST_HEADER)


def test_text_prompt():
    assert build_text_prompt("x") == USER_HEADER + "x" + ASSIST_HEADER


# ---- session ----------------------------------------------------------------


def test_session_roundtrip():
    s = Session()
    s.history.append(Turn("q", "a"))
    again = Session.from_dict(s.to_dict())
    assert again.session_id == s.session_id and again.history == s.history
    s.reset()
    assert s.history == []


# ---- generation config ------------------------------------------------------


def test_generation_config_validation():
    with pytest.raises(ValueError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(ValueError):
        GenerationConfig(top_p=0)
    assert GenerationConfig().resolved(99).max_new_tokens == 99
    assert GenerationConfig(max_new_tokens=5).resolved(99).max_new_tokens == 5


# ---- auth -------------------------------------------------------------------


def test_token_explicit_wins(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert resolve_token("explicit") == "explicit"


def test_token_from_env(monkeypatch):
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert resolve_token() == "from_env"


def test_empty_explicit_token_rejected():
    with pytest.raises(AuthenticationError):
        resolve_token("   ")


def test_mask_token():
    assert "abcdefghijkl" not in mask_token("abcdefghijkl")
    assert mask_token(None) == "<none>"


# ---- images -----------------------------------------------------------------


def _png_bytes():
    buf = io.BytesIO()
    Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_load_image_from_bytes_pil_path_and_filelike(tmp_path):
    data = _png_bytes()
    assert load_image(data).mode == "RGB"
    assert load_image(Image.new("L", (4, 4))).mode == "RGB"
    p = tmp_path / "x.png"
    p.write_bytes(data)
    assert load_image(str(p)).size == (8, 8)
    assert load_image(p).size == (8, 8)
    assert load_image(io.BytesIO(data)).size == (8, 8)


def test_remote_urls_can_be_disabled():
    with pytest.raises(ImageLoadError):
        load_image("http://example.com/a.jpg", allow_remote=False)


def test_bad_inputs():
    with pytest.raises(ImageLoadError):
        load_image(b"not an image")
    with pytest.raises(ImageLoadError):
        load_image("/nope/missing.jpg")
    with pytest.raises(ImageLoadError):
        load_image(12345)


# ---- chat cascade (stubbed model) -------------------------------------------


def _stub_client():
    c = TestAtlasVision.__new__(TestAtlasVision)  # skip __init__: no weights
    c.calls = []

    def text(prompt, *, config=None):
        c.calls.append(("text", prompt, config))
        return "TRANSLATED"

    def ask(image, question, *, session=None, config=None):
        c.calls.append(("ask", question))
        return "two cats"

    c.text, c.ask = text, ask
    return c


def test_chat_english_does_not_translate():
    c = _stub_client()
    r = c.chat("img", "How many?", lang="en")
    assert r.answer == "two cats" and [k for k, *_ in c.calls] == ["ask"]


def test_chat_igbo_runs_full_cascade():
    c = _stub_client()
    r = c.chat("img", "Kedu ihe dị na foto a?", lang="Igbo")
    assert [k for k, *_ in c.calls] == ["text", "ask", "text"]
    assert r.language == "ig" and r.answer == "TRANSLATED" and r.english_answer == "two cats"
    # the back-translation gets a larger token budget than the default
    assert c.calls[2][2].max_new_tokens == 512


def test_back_translation_gets_double_an_explicit_budget():
    """Regression: with max_new_tokens=80 the answer was cut off mid-sentence in Yoruba/Hausa."""
    from testatlasvision import GenerationConfig

    c = _stub_client()
    c.chat("img", "Kí ni ó wà nínú àwòrán yìí?", lang="yo", config=GenerationConfig(max_new_tokens=80))
    kinds = [k for k, *_ in c.calls]
    assert kinds == ["text", "ask", "text"]
    assert c.calls[0][2].max_new_tokens == 80  # question translation: as given
    assert c.calls[2][2].max_new_tokens == 160  # answer translation: doubled


def test_chat_skip_question_translation():
    c = _stub_client()
    c.chat("img", "How many?", lang="yo", translate_question=False)
    assert [k for k, *_ in c.calls] == ["ask", "text"]


def test_translate_same_language_is_noop():
    c = _stub_client()
    assert c.translate("hello", "en", "English") == "hello" and c.calls == []


def test_import_does_not_pull_in_heavy_libs():
    """`import testatlasvision` must stay cheap so the API layer can import types anywhere."""
    import subprocess
    import sys

    code = "import testatlasvision, sys; bad = [m for m in ('torch', 'transformers', 'peft') if m in sys.modules]; sys.exit(1 if bad else 0)"
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


# ---- gated-model access mapping (Hub calls faked) ---------------------------


def _hub_error(cls, status):
    from unittest.mock import Mock

    return cls("boom", response=Mock(status_code=status, headers={}))


@pytest.mark.parametrize(
    "exc_name,status,token,expected",
    [
        ("GatedRepoError", 403, None, "AuthenticationError"),
        ("GatedRepoError", 401, "hf_bad", "AuthenticationError"),
        ("GatedRepoError", 403, "hf_ok_no_access", "GatedModelAccessError"),
        ("RepositoryNotFoundError", 401, "hf_bad", "AuthenticationError"),
        ("RepositoryNotFoundError", 404, "hf_ok", "ModelLoadError"),
    ],
)
def test_ensure_llm_access_error_mapping(monkeypatch, exc_name, status, token, expected):
    import huggingface_hub
    import huggingface_hub.utils as hu

    from testatlasvision import auth, exceptions

    def fake_check(*a, **k):
        raise _hub_error(getattr(hu, exc_name), status)

    monkeypatch.setattr(huggingface_hub, "auth_check", fake_check, raising=False)
    with pytest.raises(getattr(exceptions, expected)):
        auth.ensure_llm_access("NCAIR1/N-ATLaS", token)


def test_ensure_llm_access_ok(monkeypatch):
    import huggingface_hub

    from testatlasvision import auth

    monkeypatch.setattr(huggingface_hub, "auth_check", lambda *a, **k: None, raising=False)
    auth.ensure_llm_access("NCAIR1/N-ATLaS", "hf_ok")  # no exception


def test_old_hub_without_auth_check_forces_a_real_server_call(monkeypatch):
    """Regression: hf_hub_download alone returns a cached file even for a fake token."""
    import huggingface_hub

    from testatlasvision import auth

    seen = {}
    monkeypatch.setattr(huggingface_hub, "auth_check", None, raising=False)  # simulate an old huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: seen.update(k) or "/x")
    auth.ensure_llm_access("NCAIR1/N-ATLaS", "hf_ok")
    assert seen.get("force_download") is True


def test_offline_does_not_block_startup(monkeypatch, caplog):
    import huggingface_hub

    from testatlasvision import auth

    def down(*a, **k):
        raise ConnectionError("no network")

    monkeypatch.setattr(huggingface_hub, "auth_check", down, raising=False)
    auth.ensure_llm_access("NCAIR1/N-ATLaS", "hf_ok")  # warns, does not raise
