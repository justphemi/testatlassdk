# Contributing

## Set up

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --no-deps -e .
pip install pytest ruff build twine pillow huggingface_hub
```

The offline tests never load a model, so you do **not** need torch or a GPU for day-to-day work.

## Before opening a pull request

```bash
pytest -q
ruff check .
ruff format .
```

CI runs the same checks on Python 3.10 to 3.13, plus a package build and `twine check`.

## Testing the model code (needs a GPU)

Model loading and generation can only be verified on a GPU. Use Google Colab (T4 is enough):

1. Open `notebooks/pypi_smoke_test.ipynb` to test the **published** release.
2. To test unreleased changes, replace its install cell with:
   `!pip install -q "testatlasvision[4bit] @ git+https://github.com/testatlassdk.git@your-branch"`
3. The last cell prints PASS/FAIL for each check.

## Releasing

1. Bump `__version__` in `src/testatlasvision/__init__.py` and add an entry to `CHANGELOG.md`.
2. Check that `pyproject.toml` has no `YOUR-ORG` left, and commit.
3. Tag it: `git tag vX.Y.Z && git push --tags`.
4. Build and check: `rm -rf dist build && python -m build && twine check dist/*`.
5. Upload: `twine upload dist/*` with username `__token__` and a PyPI token **scoped to this project**.

Published versions can never be replaced or edited, so every fix needs a new version number.

Never commit tokens (PyPI or Hugging Face). If one leaks into a chat, a log or a commit, revoke it immediately.
