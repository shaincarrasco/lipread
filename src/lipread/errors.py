"""Exception types raised by LipRead."""
from __future__ import annotations


class LipReadError(RuntimeError):
    """A user-facing error: bad input, missing model, licence not accepted, or a failed stage."""
