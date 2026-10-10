# testatlasvision

[![PyPI](https://img.shields.io/pypi/v/testatlasvision)](https://pypi.org/project/testatlasvision/)
[![CI](https://github.com/testatlassdk/actions/workflows/ci.yml/badge.svg)](https://github.com/testatlassdk/actions/workflows/ci.yml)
[![Open the tutorial in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/testatlassdk/blob/main/notebooks/atlasvision_tutorial.ipynb)

Python SDK for **AtlasVision**: ask questions about images in **English, Igbo, Yoruba and Hausa**.

AtlasVision combines a SigLIP2 vision encoder, the N-ATLaS language model and trained projector/LoRA weights.
This package loads those pieces and gives you a small, typed API on top.

> **Needs a GPU** (a free Colab T4 is enough) and a Hugging Face account with access to the gated
> N-ATLaS model. It will not run on an ordinary laptop. The first load downloads about 16 GB.

## Try it first

Open the **[interactive tutorial in Colab](https://colab.research.google.com/github/testatlassdk/blob/main/notebooks/atlasvision_tutorial.ipynb)**.
It walks through installing, loading the model, asking questions in all four languages and holding a conversation about an image.

## Install

```bash
pip install testatlasvision
pip install "testatlasvision[4bit]"   # adds bitsandbytes: needed on GPUs with under 20 GB (e.g. Colab T4)
```

Requires Python 3.10+.

## Hugging Face access

The base model, N-ATLaS, is gated.

1. Request access at <https://huggingface.co/NCAIR1/N-ATLaS> and wait for approval.
2. Create a token at <https://huggingface.co/settings/tokens>.
   A **Read** token is the simplest. For a shared server, prefer a **fine-grained** token with
   *"Read contents of public gated repos you can access"* ticked (and read access to the weights repo if it is private).
   The SDK only downloads weights, so never use a Write or Full Access token.
3. Give the token to the SDK in any one of these ways (checked in this order):

| Where | How |
|---|---|
| Code | `TestAtlasVision(token="hf_...")` |
| Environment | `export HF_TOKEN=hf_...` |
| Colab | A Secret named `HF_TOKEN` with *Notebook access* on |
| Cached login | `huggingface-cli login` |
| Prompt | `TestAtlasVision(interactive_login=True)` |

Before the large downloads the client checks access and raises a clear error:
`AuthenticationError` (missing or invalid token) or `GatedModelAccessError` (token fine, access not granted).
The token is never logged (only a masked form) and never written to disk by the SDK.

## Quick start

```python
from testatlasvision import TestAtlasVision, GenerationConfig

model = TestAtlasVision()  # load once; takes minutes

model.ask("cats.jpg", "How many animals are in this image?")  # -> str
model.describe("cats.jpg", lang="yo")  # caption in Yoruba

r = model.chat("cats.jpg", "Kedu ihe dị na foto a?", lang="ig")  # -> ChatResult
r.answer, r.english_question, r.english_answer

# Follow-up questions
session = model.new_session()
model.ask("cats.jpg", "What is the main object?", session=session)
model.ask("cats.jpg", "What colour is it?", session=session)

# Text only (no image), and translation
model.text("Explain what a positron is.")
model.translate("Good morning", "ha")

# Decoding settings
model.ask("cats.jpg", "Describe it", config=GenerationConfig(max_new_tokens=100, temperature=0.7))
```

`image` can be a PIL image, a file path, raw `bytes`, a file-like object or an `http(s)` URL.

## Languages

Supported codes: `en`, `ig` (Igbo), `yo` (Yoruba), `ha` (Hausa). Names such as `"Igbo"` also work.

The vision model answers in English. For non-English questions the SDK runs a cascade:
translate the question to English, answer, translate the answer back. That is up to three generations,
so expect roughly three times the latency. `ChatResult` always includes `english_question` and `english_answer`,
so you can show or check the intermediate English. Translation quality varies by language; inspect `english_answer`
if a translated answer looks off.

## API reference

| Method | Returns | Notes |
|---|---|---|
| `TestAtlasVision(token, repo, llm, vision, stage, load_in_4bit, device, interactive_login, allow_remote_images, check_access)` | client | `stage`: `"2b"` (default, recommended), `"2"`, `"1"` (captioner) |
| `ask(image, question, *, session=None, config=None)` | `str` | English only. Pass a `Session` for follow-ups |
| `chat(image, question, *, lang="en", session=None, translate_question=True, config=None)` | `ChatResult` | Question and answer in `lang` |
| `describe(image, *, lang="en", detailed=True, config=None)` | `str` | Caption |
| `text(prompt, *, config=None)` | `str` | Plain N-ATLaS, no image |
| `translate(text, target, source="en", *, config=None)` | `str` | Between `en`/`ig`/`yo`/`ha` |
| `new_session()` | `Session` | Conversation memory, always stored in English; `to_dict()` / `from_dict()` for storage |
| `info()` | `ModelInfo` | Handy for a health endpoint |
| `close()` | `None` | Free GPU memory |

### Errors

All errors subclass `AtlasVisionError`.

| Exception | Meaning | Suggested HTTP status |
|---|---|---|
| `AuthenticationError` | No or invalid Hugging Face token | 500 (server misconfigured) |
| `GatedModelAccessError` | Token valid, no access to the gated model | 500 |
| `ModelLoadError` | Weights failed to load | 503 |
| `ImageLoadError` | Image could not be read or was rejected | 400 / 422 |
| `UnsupportedLanguageError` | Unknown language | 422 |

## Building a service on top

- **Load once** at startup and keep one client per GPU. Loading takes minutes.
- **Sessions replace shared history.** The client is stateless: store `Session.to_dict()` per user (Redis or a database) keyed by `session_id`.
- **Generation is serialised** by an internal lock (one GPU, and the LoRA toggle in `text()` is global). Concurrent requests queue; scale with more workers or GPUs, not threads.
- **Latency:** on a Colab T4, an English question takes about 3 seconds and a non-English chat roughly 15 to 20. Use generous timeouts or an async job pattern.
- **Public endpoints:** create the client with `allow_remote_images=False` (fetching arbitrary URLs is an SSRF risk) and accept uploads instead. Downloads are capped at 25 MB.
- **The server needs its own token** (`HF_TOKEN`). End users never need one.
- `import testatlasvision` does not import torch, so request/response types can be shared with code that runs without a GPU.

## Limitations

- GPU required; no streaming or batching yet.
- Model loading and generation are verified manually on Colab (see `notebooks/pypi_smoke_test.ipynb`). The automated tests cover everything that does not need weights.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

```bash
pip install --no-deps -e . && pip install pytest ruff pillow huggingface_hub
pytest -q
```

## License

TODO: add a license before publishing.
