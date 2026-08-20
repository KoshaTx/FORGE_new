"""Checkout and verification support for FORGE's hash-pinned source data.

This is repository tooling, not part of the scientific ``forge`` API.  It is packaged only so the
installed ``forge data`` command works from any checkout location.
"""

from forge_data.vendor import main

__all__ = ["main"]
