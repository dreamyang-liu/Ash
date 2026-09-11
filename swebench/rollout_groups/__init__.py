"""Stable Ash rollout interface used by Miles and rollout strategies.

The HTTP service in :mod:`swebench.rollout_groups.server` deliberately knows
nothing about how a branch is selected. A strategy receives a validated
rollout-group request and uses the model and environment contracts exported
here.
"""

from .protocol import (
    EnvironmentRef,
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
from .environment_catalog import EnvironmentCatalog, EnvironmentCatalogEntry
from .environment_resolver import (
    AgentEnvOCIResolver,
    AgentEnvOCIResolverConfig,
    AgentEnvResourceProfile,
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
    "EnvironmentCatalog",
    "EnvironmentCatalogEntry",
    "EnvironmentProvider",
    "AgentEnvOCIResolver",
    "AgentEnvOCIResolverConfig",
    "AgentEnvResourceProfile",
    "EnvironmentRef",
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
