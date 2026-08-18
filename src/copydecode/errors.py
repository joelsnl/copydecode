"""Exception hierarchy.

Catch ``CopydecodeError`` to handle every failure this package raises on
purpose; anything else escaping the API is a bug and should surface with a
traceback.
"""

from __future__ import annotations


class CopydecodeError(Exception):
    """Base class for all errors raised deliberately by copydecode."""


class DocumentError(CopydecodeError):
    """A document could not be read or written (bad file, missing optional dependency)."""
