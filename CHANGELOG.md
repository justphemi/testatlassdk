# Changelog

All notable changes to this project are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.1.1]

### Added
- `py.typed` marker (the package ships type hints).
- Public README, CONTRIBUTING guide, GitHub Actions CI, tutorial and smoke-test notebooks.

### Changed
- Version now has a single source: `testatlasvision.__version__`.
- Project metadata: classifiers, URLs, development extras.

## [0.1.0]

### Added
- First release: `TestAtlasVision` client with `ask`, `chat`, `describe`, `text`, `translate`, sessions for
  follow-up questions, typed results, and a clear error hierarchy.
- Hugging Face token resolution and a gated-model access check (`AuthenticationError`, `GatedModelAccessError`).
- Answer translation gets double the caller's token budget, so Yoruba/Hausa answers are no longer cut off.
