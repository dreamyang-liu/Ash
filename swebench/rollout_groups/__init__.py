"""Stable Ash rollout interface used by Miles and rollout strategies.

The HTTP service in :mod:`swebench.rollout_groups.server` deliberately knows
nothing about how a branch is selected. A strategy receives a validated
rollout-group request and may use Ash checkpoint/cache primitives through the
small protocols exported here.
"""

from .protocol import (
    GeneratedSpan,
    RolloutBudget,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutSubmission,
    SampleSlot,
    Trajectory,
)
from .runner import (
    EnvironmentProvider,
    GroupRolloutService,
    ModelClient,
    RolloutContext,
    RolloutStrategy,
)
from .strategies import SequentialRolloutStrategy
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
    "SampleSlot",
    "SequentialRolloutStrategy",
    "Trajectory",
    "serve",
]
