"""Model routing + per-slot tokens.

Two lookups, deliberately separate:

- ``RoutingTable`` maps the model name an agent asks for to an upstream
  (base URL + credential + optional rename). This is the model-swap mechanism:
  ``claude-sonnet-4-6 -> your vLLM`` is a table entry, not a code change.
- ``SlotToken`` maps the credential an agent presents to an ``agent_id`` and a
  budget. The orchestrator mints one per slot, so attribution and enforcement do
  not depend on the agent cooperating.

The table is plain JSON so it can be version-controlled with a run config::

    {
      "routes": {
        "default":           {"base_url": "https://api.anthropic.com",
                              "api_key_env": "ANTHROPIC_API_KEY"},
        "ash-rl-ckpt-42":    {"base_url": "http://10.0.0.5:8000",
                              "upstream_model": "checkpoint-42",
                              "api_key": "unused"}
      }
    }
"""

from __future__ import annotations

import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

DEFAULT_ROUTE = "default"


@dataclass
class ModelRoute:
    """Where a requested model actually goes."""

    base_url: str = "https://api.anthropic.com"
    #: Literal key, or ``api_key_env`` to read one from the environment. Keeping
    #: the credential here means the agent never holds a provider key -- it only
    #: ever sees its own slot token.
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    #: Rewrite the model name on the way out (agent asks X, upstream wants Y).
    upstream_model: Optional[str] = None
    #: Extra headers to add upstream (e.g. a gateway of your own downstream).
    headers: Dict[str, str] = field(default_factory=dict)

    def resolve_key(self) -> Optional[str]:
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelRoute":
        return cls(
            base_url=payload.get("base_url") or "https://api.anthropic.com",
            api_key=payload.get("api_key"),
            api_key_env=payload.get("api_key_env"),
            upstream_model=payload.get("upstream_model"),
            headers=dict(payload.get("headers") or {}),
        )


@dataclass
class SlotToken:
    """A minted credential identifying one slot, with its budget."""

    token: str
    agent_id: str
    run_id: Optional[str] = None
    #: Hard ceiling in USD; None means unlimited. Enforced before forwarding.
    budget_usd: Optional[float] = None
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0
    blocked: int = 0

    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.spent_usd >= self.budget_usd

    def as_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "run_id": self.run_id,
            "budget_usd": self.budget_usd,
            "spent_usd": round(self.spent_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "requests": self.requests,
            "blocked": self.blocked,
        }


class RoutingTable:
    """Routes + slot tokens. Thread-safe; the server is multi-threaded."""

    def __init__(self, routes: Optional[Dict[str, ModelRoute]] = None) -> None:
        self._routes: Dict[str, ModelRoute] = dict(routes or {})
        self._routes.setdefault(DEFAULT_ROUTE, ModelRoute(api_key_env="ANTHROPIC_API_KEY"))
        self._tokens: Dict[str, SlotToken] = {}
        self._lock = threading.Lock()

    # --- routes ------------------------------------------------------------
    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "RoutingTable":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        routes = {
            name: ModelRoute.from_dict(entry)
            for name, entry in (payload.get("routes") or {}).items()
        }
        return cls(routes)

    def route_for(self, model: Optional[str]) -> ModelRoute:
        """Exact model match, else ``default``."""
        if model and model in self._routes:
            return self._routes[model]
        return self._routes[DEFAULT_ROUTE]

    def add_route(self, model: str, route: ModelRoute) -> None:
        with self._lock:
            self._routes[model] = route

    def models(self) -> list:
        return sorted(m for m in self._routes if m != DEFAULT_ROUTE)

    # --- slot tokens -------------------------------------------------------
    def mint(
        self,
        agent_id: str,
        *,
        run_id: Optional[str] = None,
        budget_usd: Optional[float] = None,
    ) -> SlotToken:
        token = SlotToken(
            token="ash-slot-" + secrets.token_urlsafe(24),
            agent_id=agent_id,
            run_id=run_id,
            budget_usd=budget_usd,
        )
        with self._lock:
            self._tokens[token.token] = token
        return token

    def lookup(self, presented: Optional[str]) -> Optional[SlotToken]:
        if not presented:
            return None
        candidate = presented.strip()
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()
        with self._lock:
            return self._tokens.get(candidate)

    def charge(
        self,
        token: SlotToken,
        *,
        cost_usd: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        with self._lock:
            token.spent_usd += cost_usd
            token.input_tokens += input_tokens
            token.output_tokens += output_tokens

    def stats(self) -> list:
        with self._lock:
            return [t.as_dict() for t in self._tokens.values()]
