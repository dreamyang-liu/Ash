"""The reviewer's explicit output envelope, shared by branching controllers."""

import json
import re


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
