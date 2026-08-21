"""Pluggable hooks for the agent loop — keep cross-cutting concerns out of run().

Three extension points, each a list the agent iterates:

    before_query(agent, conv) -> None
        Runs before every model call; may mutate the Conversation in place.
    before_finish(agent, conv) -> bool
        Runs when the agent has stopped calling tools and the loop is about to
        report ``completed``. Returning True keeps it going for another turn.
    result_processor(content, name, args, result) -> str
        Runs on each tool result's content; returns the (possibly rewritten) text.

Override agent.before_query_hooks / agent.before_finish_hooks /
agent.result_processors to customize.

`before_finish` exists because the two ways a run ends need different
interventions and only one of them is predictable. A budget ceiling can be seen
coming, so `before_query` handles it; an agent deciding it is done cannot, and it
is the common case -- 24 of 25 recorded runs ended that way. Anything that must
happen before the last turn therefore needs both, which is why the hook can veto
the ending rather than merely observe it.

Scope: these hooks sit on the *model* path. Tool-path concerns moved to L2
interceptors (`interceptors.py`): output truncation used to be the one default
`result_processor` here and is now `TruncateInterceptor`, which also covers
agents reaching the sandbox through the MCP proxy. `result_processors` survives
as an extension point — it is the only place that sees the loop's
`Error:`-prefixed content — but ships empty, and the `truncate` processor is
gone rather than deprecated: running it alongside the interceptor would elide
the interceptor's own marker text.
"""


def budget_warning(agent, conv):
    """Inject a one-time budget warning when few steps/cost remain."""
    if agent._warned:
        return
    w = agent.cost.budget_warning(agent.config.step_limit, agent.config.cost_limit)
    if not w:
        return
    agent._warned = True
    conv.append_to_last(w)
    agent._trace(f"\n{w}\n")


DEFAULT_BEFORE_QUERY = [budget_warning]
#: Nothing by default: an agent that says it is finished is finished, unless a
#: caller has something to ask for first (see ``swebench/submission.py``).
DEFAULT_BEFORE_FINISH = []
DEFAULT_RESULT_PROCESSORS = []
