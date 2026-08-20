"""Pluggable hooks for the agent loop — keep cross-cutting concerns out of run().

Two extension points, each a list the agent iterates:

    before_query(agent, conv) -> None
        Runs before every model call; may mutate the Conversation in place.
    result_processor(content, name, args, result) -> str
        Runs on each tool result's content; returns the (possibly rewritten) text.

Override agent.before_query_hooks / agent.result_processors to customize.

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
DEFAULT_RESULT_PROCESSORS = []
