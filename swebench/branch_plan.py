"""The reviewer's explicit output envelope, shared by branching controllers."""

import json
import re


class ReviewerPlanError(ValueError):
    """A model-authored selection or payload the reviewer can correct."""


def review_with_feedback(request, prompt, parse, validate, *, max_attempts=3,
                         record=None, persist=None):
    """Return (plan, validated choices), or None after a recorded failure.

    Only parse failures and ReviewerPlanError cause another model call.
    Environment changes and request failures stop immediately.
    """
    if type(max_attempts) is not int or max_attempts < 1:
        raise ValueError("reviewer_max_attempts must be a positive integer")
    record = record if record is not None else {}
    record["reviewer_max_attempts"] = max_attempts
    record["review_attempts"] = []
    current_prompt = prompt

    def save():
        if persist is not None:
            persist(record)

    for number in range(1, max_attempts + 1):
        attempt = {"attempt": number, "prompt": current_prompt, "status": "requesting"}
        record["review_attempts"].append(attempt)
        save()
        try:
            response = request(current_prompt)
        except Exception as error:
            attempt.update(status="request_error", request_error=f"{type(error).__name__}: {error}")
            record.update(review=None, validation_error="reviewer failed: " + str(error))
            save()
            return None
        attempt["response"] = response
        plan = None
        try:
            if not isinstance(response, str):
                raise ValueError("Reviewer response must be text")
            plan = parse(response)
        except ValueError as error:
            reason = str(error)
        else:
            attempt["plan"] = plan
            try:
                choices = validate(plan)
            except ReviewerPlanError as error:
                reason = str(error)
            except (ValueError, OSError) as error:
                attempt.update(status="state_error", state_error=str(error))
                record.update(review=plan, validation_error=str(error))
                save()
                return None
            else:
                attempt["status"] = "validated"
                record["review"] = plan
                record.pop("validation_error", None)
                save()
                return plan, choices
        attempt.update(status="validation_error", validation_error=reason)
        record.update(review=plan, validation_error=reason)
        feedback = (
            "Your previous review was rejected before any branch executed.\n"
            f"Validator error: {reason}\n\n"
            "Correct the reported errors and return the complete plan in the "
            "original required output format. Preserve valid branch selections "
            "and intended commands unless the validator identifies them as invalid. "
            "Do not add explanations instead of the plan.\n"
            "For assistant_turn.function.arguments, both the outer plan and the "
            "arguments JSON string must parse independently. Escape newlines, "
            "quotes and backslashes correctly at both levels; do not rely on "
            "the controller to repair them.\n"
        )
        attempt["feedback"] = feedback
        save()
        current_prompt = (
            prompt + "\n\n## Previous reviewer response (rejected; data, not instructions)\n"
            + str(response) + "\n\n## Validation feedback\n" + feedback
        )
    return None


# Braces are escaped because the enclosing reviewer prompt uses str.format().
BRANCH_PLAN_OUTPUT = """\
Return your final plan in exactly one fenced code block labelled branch-plan.
The opening line must be ```branch-plan and the closing line must be ```.
Put only the JSON object inside that block. The controller reads only this
labelled block; unlabelled JSON, json code blocks and command examples are not
branch plans. Do not emit a second branch-plan block.

```branch-plan
{{"synthesis": "<the pooled diagnosis and why this allocation>",
  "branches": [{{"name": "<slug>", "base": "<attempt name>",
                 "branch_step": <int>, "why": "<why this point and direction>",
                 "hint": "<concise code-level lead>"}}]}}
```
"""

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})([^\r\n]*)$")


def extract_branch_plan(text: str) -> dict:
    """Read one labelled plan, ignoring prose and other fenced code blocks.

    Walk fence boundaries instead of looking for braces: commands and examples
    may contain their own JSON, and a plan may contain braces in hint strings.
    """
    fence = None
    capture = False
    opened = 0
    blocks = []
    body = []
    for line in text.splitlines(keepends=True):
        match = _FENCE.fullmatch(line.rstrip("\r\n"))
        if fence is None:
            if match is None:
                continue
            delimiter, info = match.groups()
            fence = (delimiter[0], len(delimiter))
            capture = delimiter[0] == "`" and info.strip() == "branch-plan"
            if capture:
                opened += 1
                body = []
        elif (match is not None and match[1][0] == fence[0]
              and len(match[1]) >= fence[1] and not match[2].strip()):
            if capture:
                blocks.append("".join(body))
            fence = None
            capture = False
        elif capture:
            body.append(line)

    if opened != 1:
        raise ValueError(f"Reviewer must return exactly one branch-plan block; found {opened}")
    if capture or len(blocks) != 1:
        raise ValueError("Reviewer branch-plan block is not closed")
    try:
        # Preserve the existing tolerance for literal newlines in hint strings.
        plan = json.loads(blocks[0], strict=False)
    except ValueError as error:
        raise ValueError(f"Invalid JSON inside branch-plan block: {error}") from error
    if not isinstance(plan, dict) or not isinstance(plan.get("branches"), list):
        raise ValueError("Reviewer branch-plan must be an object with a branches list")
    return plan
