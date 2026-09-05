"""Stable group-rollout boundary used by Miles and Ash strategies.

The HTTP service in :mod:`swebench.rollout_groups.server` deliberately knows
nothing about how a branch is selected.  A strategy receives a validated group
request and may use the checkpoint/cache primitives from ``ash_sandbox`` (or a
future implementation) through the small protocols exported here.
"""

from .protocol import (
    GeneratedSpan,
    RolloutBudget,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutSubmission,
    Trajectory,
)
from .runner import (
    EnvironmentProvider,
    GroupRolloutService,
    ModelClient,
    RolloutContext,
    RolloutStrategy,
)
from .server import RolloutGroupsHTTPServer, serve

__all__ = [
    "EnvironmentProvider",
    "GeneratedSpan",
    "GroupRolloutService",
    "ModelClient",
    "RolloutBudget",
    "RolloutContext",
    "RolloutGroupRequest",
    "RolloutGroupResult",
    "RolloutGroupsHTTPServer",
    "RolloutStrategy",
    "RolloutSubmission",
    "Trajectory",
    "serve",
]
