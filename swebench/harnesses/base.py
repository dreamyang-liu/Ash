"""Base class for agent harnesses."""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseHarness(ABC):
    """Abstract base for SWE-bench agent harnesses.

    A harness takes an instance + config and returns a prediction dict:
      {"instance_id": ..., "model_patch": ..., "model_name_or_path": ..., "exit_status": ...}
    """

    def __init__(self, config: dict):
        """Initialize with a merged config dict (from YAML + CLI)."""
        self.config = config

    @abstractmethod
    def run_instance(self, instance: dict, output_dir: Path) -> dict:
        """Run the agent on a single instance. Returns prediction dict."""
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
