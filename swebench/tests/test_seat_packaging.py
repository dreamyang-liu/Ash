"""One package per seat — the structure, asserted.

A seat is the unit somebody replaces, so each gets a directory: a caller writing
different truncation, a different rendering, or a different rule adds a package
beside these rather than editing a file shared with the seats they are keeping.

These tests pin the boundaries that make that true, because they are the kind that
erode quietly -- the next seat gets appended to whichever file is open.
"""

from __future__ import annotations

import pathlib

import pytest

SEATS_DIR = pathlib.Path(__file__).resolve().parents[1] / "agent" / "seats"


def _seat_packages():
    return sorted(p for p in SEATS_DIR.iterdir()
                  if p.is_dir() and not p.name.startswith("__"))


def test_every_seat_is_its_own_package():
    """Not a module in a shared file: a directory, so it has room to grow the
    state, predicates and renderers its rule needs."""
    names = [p.name for p in _seat_packages()]
    assert set(names) == {"guardrail", "truncate", "present"}, names
    for pkg in _seat_packages():
        assert (pkg / "__init__.py").exists(), f"{pkg.name} is not importable"
        assert (pkg / "interceptor.py").exists(), \
            f"{pkg.name} has no interceptor.py; where is the seat?"


def test_each_seat_exports_its_interceptor():
    import importlib

    expected = {"guardrail": "GuardrailInterceptor",
                "truncate": "TruncateInterceptor",
                "present": "OutcomePresenter"}
    for pkg, cls in expected.items():
        module = importlib.import_module(f"swebench.agent.seats.{pkg}")
        assert hasattr(module, cls), f"{pkg} does not export {cls}"


def test_no_seat_imports_another_seat():
    """Seats are peers. One reaching into another would make the pair a unit, and
    replacing either would mean understanding both."""
    offenders = []
    for pkg in _seat_packages():
        others = {p.name for p in _seat_packages()} - {pkg.name}
        for path in pkg.rglob("*.py"):
            source = path.read_text()
            for other in others:
                if f"from ..{other}" in source or f"seats.{other}" in source:
                    offenders.append(f"{path.name} imports {other}")
    assert offenders == [], offenders


def test_the_assembly_is_not_inside_a_seat():
    """default_pipeline states the order, which is a fact about the three seats
    together -- putting it in one of them would make that seat special."""
    for pkg in _seat_packages():
        for path in pkg.rglob("*.py"):
            assert "def default_pipeline" not in path.read_text(), \
                f"the assembly leaked into {pkg.name}/{path.name}"
    assert "def default_pipeline" in (SEATS_DIR / "pipeline.py").read_text()


def test_the_metadata_keys_belong_to_the_pipeline_not_a_seat():
    """RAW_OUTPUT / RAW_ERROR are the protocol seats use to talk to the host, so
    they live with the pipeline. They used to be re-exported by the seat module,
    which is how the agent loop came to import them from there."""
    from swebench.agent import pipeline, seats

    assert hasattr(pipeline, "RAW_OUTPUT") and hasattr(pipeline, "RAW_ERROR")
    assert not hasattr(seats, "RAW_OUTPUT"), \
        "re-exporting these invites importing them from the wrong layer"


def test_seats_are_reachable_from_one_place_for_callers_who_want_the_set():
    """A caller composing a chain should not need three imports."""
    from swebench.agent.seats import (EDIT_STREAK_LIMIT, GuardrailInterceptor,
                                      GuardrailState, OutcomePresenter,
                                      TEST_MARKERS, TruncateInterceptor,
                                      default_pipeline, render_outcome)
    assert default_pipeline().interceptors      # smoke: the set composes
