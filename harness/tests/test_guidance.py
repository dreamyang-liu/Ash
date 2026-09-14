import pytest

from harness.core.guidance import render_branch_note


@pytest.mark.parametrize("hint", ["", "  \n", None, {"hint": "inspect"}])
def test_hint_requires_nonempty_text(hint):
    with pytest.raises(ValueError, match="non-empty string"):
        render_branch_note(hint)


@pytest.mark.parametrize("truncated", [True, False])
@pytest.mark.parametrize("commit_required", [True, False])
def test_hint_body_is_preserved_exactly(truncated, commit_required):
    hint = (
        "  The caller owns /environments/:id/compliance.\n"
        "RegisterRoutes registers only /baselines beneath that group.\n"
        "Check both component registration and the bootstrap caller.  "
    )
    note = render_branch_note(hint, truncated=truncated,
                              commit_required=commit_required)
    assert note.count(hint) == 1
    assert ("restored to an earlier checkpoint" in note) is not truncated
    assert ("Commit your changes" in note) is commit_required
    assert "system-reminder" not in note


def test_renderer_does_not_filter_or_rewrite_reviewer_words():
    hint = "Inspect TestPublicCase in public_test.go."
    assert hint in render_branch_note(hint)


def test_true_fork_note_starts_with_the_lead_and_requests_direct_continuation():
    hint = "Check whether the projection retains accepted aliases."
    note = render_branch_note(hint)
    assert note.startswith(hint + "\n\n")
    assert "without acknowledging or recapping this message" in note
    assert "do not invent\nprior observations" in note
    for meta in ("system-reminder", "Before finalizing", "reviewer", "feedback"):
        assert meta not in note


def test_hint_note_keeps_restoration_caveat_without_private_diagnostics():
    note = render_branch_note("Inspect the current code and reproduce the boundary case.",
                              truncated=False, commit_required=True)
    assert "restored to an earlier checkpoint" in note
    assert "Commit your changes" in note
    assert "grader" not in note and "verdict" not in note
