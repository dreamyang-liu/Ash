import pytest

from swebench import fork_eval
from deepswe.bench import DeepSWE, branch_note


@pytest.mark.parametrize("benchmark", ["swebench", "deepswe"])
@pytest.mark.parametrize("truncated", [True, False])
def test_branch_never_receives_verdict_or_analyst_report(tmp_path, benchmark, truncated):
    bench = fork_eval.SweBench() if benchmark == "swebench" else DeepSWE(tmp_path)
    hint = "Inspect the relevant code and reproduce the issue."
    note = bench.branch_prompt(
        {"problem": "PUBLIC_TASK", "f2p": ["TestPrivate"], "p2p": []},
        verdict="PRIVATE_VERDICT", hint=hint,
        truncated=truncated, grade=fork_eval.Grade(patch="PRIVATE_PATCH"),
        analysis={"failure_reason": "PRIVATE_DIAGNOSIS", "lesson": "PRIVATE_LESSON"})
    for private in ("PRIVATE_VERDICT", "PRIVATE_PATCH", "PRIVATE_DIAGNOSIS",
                    "PRIVATE_LESSON", "TestPrivate"):
        assert private not in note
    assert "PUBLIC_TASK" not in note
    assert note.count(hint) == 1
    assert ("restored to an earlier checkpoint" in note) is not truncated


def test_legacy_deepswe_note_does_not_restore_feedback_payload():
    hint = "Inspect the current registration boundary."
    note = branch_note(71, fork_eval.Grade(detail="PRIVATE_RESULT"),
                       {"failure_reason": "PRIVATE_DIAGNOSIS"}, hint)
    assert hint in note
    assert "PRIVATE_RESULT" not in note and "PRIVATE_DIAGNOSIS" not in note
