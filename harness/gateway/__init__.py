"""Inference gateway: the model seam.

An agent's traffic reaches the provider through here, which buys three things a
slot cannot give us:

1. **Model swap** -- the reason this exists. Point ``claude-code`` at a different
   endpoint (another provider, a local vLLM, an RL checkpoint being trained)
   without touching the agent. Eval matrices and RL rollout both need it, and no
   SDK event stream can substitute.
2. **Wire-level tap** -- what the model actually received and returned, including
   the final system prompt and exact token counts, independent of whatever the
   agent chooses to report.
3. **Budget hard-stop** -- refuse requests once a run is over budget. Slot-side
   accounting can only ask an agent to stop; this can make it stop.

Not a policy engine: tool-call rewriting stays in the MCP proxy (L2), where the
call is semantically addressable. This layer speaks HTTP and tokens.

Wiring is a single environment variable, which is why it works for any agent::

    ANTHROPIC_BASE_URL=http://127.0.0.1:8787
    ANTHROPIC_AUTH_TOKEN=<per-slot token minted by the gateway>

See harness/gateway/README.md for the routing table format and the protocol
notes (why headers are forwarded verbatim, why /v1/models must answer).
"""

from harness.gateway.routing import ModelRoute, RoutingTable, SlotToken
from harness.gateway.server import GatewayServer, serve

__all__ = ["ModelRoute", "RoutingTable", "SlotToken", "GatewayServer", "serve"]
