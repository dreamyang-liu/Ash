"""One file per interceptor — the structure, asserted.

An interceptor is the unit somebody replaces, so each gets its own module: a
caller writing different truncation, a different rendering, or a rule of their own
adds a file beside these rather than appending to one shared with the interceptors
they keep.

A directory apiece was the earlier convention. It was more structure than these
need -- guardrail was the only one with enough parts to fill one, and three
packages of two files each made the set look more elaborate than it is.

The list is **not a taxonomy**: these are the interceptors we happened to need, and
a new one does not have to fit the categories. What these tests pin is that each
stays independently replaceable, because that is what erodes quietly -- the next
interceptor gets appended to whichever file is already open.
"""

from __future__ import annotations

import importlib
import pathlib

SEATS_DIR = (pathlib.Path(__file__).resolve().parents[2]
             / "harness" / "execution" / "interceptors")

#: Every interceptor module and the class it must export. Adding an interceptor
#: means adding a line here -- deliberately, so the structural rules below cover
#: it too.
INTERCEPTORS = {
    "guardrail": "GuardrailInterceptor",
    "truncate": "TruncateInterceptor",
    "present": "OutcomePresenter",
    "mutation": "MutationTracker",
}

#: Not interceptors: the assembly, and the package's own re-exports.
NON_INTERCEPTOR_MODULES = {"pipeline", "__init__"}


def _modules():
    return sorted(p for p in SEATS_DIR.glob("*.py") if p.stem != "__init__")


def test_every_interceptor_is_its_own_module():
    """One file each, and no directories: a package here would be structure the
    interceptors do not need."""
    assert not [p for p in SEATS_DIR.iterdir()
                if p.is_dir() and not p.name.startswith("__")], \
        "an interceptor grew a directory again"
    stems = {p.stem for p in _modules()}
    assert stems == set(INTERCEPTORS) | (NON_INTERCEPTOR_MODULES - {"__init__"}), stems


def test_each_module_exports_its_interceptor():
    for module_name, cls in INTERCEPTORS.items():
        module = importlib.import_module("harness.execution.interceptors.%s" % module_name)
        assert hasattr(module, cls), "%s does not export %s" % (module_name, cls)


def test_no_interceptor_imports_another_interceptor():
    """Interceptors are peers. One reaching into another would make the pair a
    unit, and replacing either would mean understanding both."""
    offenders = []
    for path in _modules():
        if path.stem in NON_INTERCEPTOR_MODULES:
            continue
        others = set(INTERCEPTORS) - {path.stem}
        source = path.read_text()
        for other in others:
            if "interceptors.%s" % other in source or "from .%s" % other in source:
                offenders.append("%s imports %s" % (path.name, other))
    assert offenders == [], offenders


def test_the_assembly_is_not_inside_an_interceptor():
    """default_pipeline states the order, which is a fact about the set together --
    putting it in one of them would make that interceptor special."""
    for path in _modules():
        if path.stem == "pipeline":
            continue
        assert "def default_pipeline" not in path.read_text(), \
            "the assembly leaked into %s" % path.name
    assert "def default_pipeline" in (SEATS_DIR / "pipeline.py").read_text()


def test_the_default_chain_is_the_three_that_shape_what_the_model_sees():
    """mutation is an interceptor but not a default one: it exists to decide
    whether a step is worth snapshotting, so it is mounted by whoever turns
    checkpointing on. Most runs do not, and it would otherwise cost every run."""
    from harness.execution.interceptors import default_pipeline

    names = [i.name for i in default_pipeline().interceptors]
    assert "MutationTracker" not in names, \
        "mutation tracking is opt-in; it should not be in the default chain"
    assert set(names) == {"GuardrailInterceptor", "TruncateInterceptor",
                          "OutcomePresenter"}, names


def test_the_metadata_keys_belong_to_the_pipeline_not_an_interceptor():
    """RAW_OUTPUT / RAW_ERROR are the protocol interceptors use to talk to the
    host, so they live with the pipeline. They used to be re-exported by the
    interceptor module, which is how a caller came to import them from
    there."""
    from harness.execution import interceptors, pipeline

    assert hasattr(pipeline, "RAW_OUTPUT") and hasattr(pipeline, "RAW_ERROR")
    assert not hasattr(interceptors, "RAW_OUTPUT"), \
        "re-exporting these invites importing them from the wrong layer"


def test_interceptors_are_reachable_from_one_place_for_callers_who_want_the_set():
    """A caller composing a chain should not need four imports."""
    from harness.execution.interceptors import (EDIT_STREAK_LIMIT, TEST_MARKERS,
                                             GuardrailInterceptor, GuardrailState,
                                             OutcomePresenter, TruncateInterceptor,
                                             default_pipeline, render_outcome)

    assert default_pipeline().interceptors      # smoke: the set composes
