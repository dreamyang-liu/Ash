"""Make ash_sandbox importable for the SDK's own tests.

Preference order matters. If the package is installed (pip install ./sdk, or a
released wheel), these tests exercise the *installed* package -- which is what
a user gets, and the only way a packaging mistake such as a module missing from
the wheel can fail a test. Only when it is absent do we fall back to the source
tree beside this directory.

Each test module used to insert the repo root itself, which quietly anchored the
SDK's tests to the harness repo: they passed from anywhere inside it and would
have broken the moment the package moved to its own repository.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: sdk/ -- the directory containing the ash_sandbox package.
SDK_ROOT = Path(__file__).resolve().parents[1]


def _ensure_importable() -> None:
    try:
        import ash_sandbox  # noqa: F401
    except ModuleNotFoundError:
        sys.path.insert(0, str(SDK_ROOT))


_ensure_importable()
