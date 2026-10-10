"""Exception hierarchy. Catch `AtlasVisionError` to handle anything the SDK raises on purpose.

The API layer can map these straight onto HTTP status codes:

    AuthenticationError        -> 401 / 500 (server misconfigured)
    GatedModelAccessError      -> 500 (server's HF account lacks access)
    ImageLoadError             -> 400 / 422
    UnsupportedLanguageError   -> 422
    ModelLoadError             -> 503
"""

from __future__ import annotations


class AtlasVisionError(Exception):
    """Base class for all SDK errors."""


class AuthenticationError(AtlasVisionError):
    """No (valid) Hugging Face token was available for a repo that needs one."""


class GatedModelAccessError(AtlasVisionError):
    """The token is valid but its account has not been granted access to the gated model."""


class ModelLoadError(AtlasVisionError):
    """Loading weights failed (missing dependency, out of memory, bad repo/stage, ...)."""


class ImageLoadError(AtlasVisionError, ValueError):
    """The image input could not be read or was rejected."""


class UnsupportedLanguageError(AtlasVisionError, ValueError):
    """Language is not one of the supported codes/names."""
