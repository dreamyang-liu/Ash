"""Stable Ash rollout interface used by Miles and rollout strategies.

The HTTP service in :mod:`swebench.rollout_groups.server` deliberately knows
nothing about how a branch is selected. A strategy receives a validated
rollout-group request and uses the model and environment contracts exported
here.
"""

from .protocol import (
    GeneratedSpan,
    RolloutBudget,
    RolloutDeletion,
    RolloutGroupRequest,
    RolloutGroupResult,
    RolloutSubmission,
    SampleSlot,
    Trajectory,
)
from .runner import (
    EnvironmentCheckpoint,
    EnvironmentProvider,
    GroupRolloutService,
    ModelClient,
    RolloutContext,
    RolloutStrategy,
)
from .session_runtime import (
    MilesSessionClient,
    SessionAgentStrategySupport,
    branch_input_length,
    trajectory_from_session,
)
from .strategies import (
    CheckpointAgentLoopRolloutStrategy,
    MilesSessionAgentRolloutStrategy,
    SequentialRolloutStrategy,
)
from .server import RolloutGroupsHTTPServer, build_service, serve

__all__ = [
    "EnvironmentCheckpoint",
    "EnvironmentProvider",
    "GeneratedSpan",
    "GroupRolloutService",
    "ModelClient",
    "MilesSessionClient",
    "CheckpointAgentLoopRolloutStrategy",
    "MilesSessionAgentRolloutStrategy",
    "RolloutBudget",
    "RolloutContext",
    "RolloutDeletion",
    "RolloutGroupRequest",
    "RolloutGroupResult",
    "RolloutGroupsHTTPServer",
    "build_service",
    "RolloutStrategy",
    "RolloutSubmission",
    "SampleSlot",
    "SessionAgentStrategySupport",
    "SequentialRolloutStrategy",
    "Trajectory",
    "branch_input_length",
    "serve",
    "trajectory_from_session",
]
