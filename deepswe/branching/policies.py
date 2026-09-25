"""Pure branch selectors. Steps identify state AFTER a completed tool turn.

These are sampling adaptations, not implementations of either paper's RL loss.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Entropy:
    lower_bound_nats: float
    top_mass: float
    tail_mass: float
    candidate_count: int


def token_entropy(token: dict) -> Entropy:
    """Shannon entropy with unreported vocabulary collapsed to ONE tail bin.

    This is a lower bound, not full-vocabulary entropy. Never normalize the
    reported top-k probabilities to one or treat missing probabilities as zero.
    """
    entries = token.get("top_logprobs")
    if not entries:
        raise ValueError("BPO requires top_logprobs; this provider omitted them")
    probabilities = {}
    for entry in entries:
        logprob = float(entry["logprob"])
        if not math.isfinite(logprob) or logprob > 0:
            raise ValueError("Non-finite or positive token log probability")
        raw = entry.get("bytes")
        identity = bytes(raw) if raw is not None else entry["token"].encode("utf-8")
        if identity in probabilities:
            raise ValueError("Duplicate token in top_logprobs")
        probabilities[identity] = math.exp(logprob)
    mass = math.fsum(probabilities.values())
    if mass > 1 + 1e-6:
        raise ValueError("Reported top-token probability mass exceeds one")
    tail = max(0., 1. - mass)
    entropy = -math.fsum(p * math.log(p) for p in [*probabilities.values(), tail] if p)
    return Entropy(entropy, mass, tail, len(probabilities))


def bpo_select(scores: list[dict], budget: int = 7, min_spacing: int = 64,
               max_points: int = 7) -> list[dict]:
    """Rank a frozen backbone; distribute the extra-rollout budget round-robin.

    A point receives K-1 new siblings, in addition to the original suffix.
    Fewer eligible separated points increases K; it never creates fake points.
    """
    if budget < 0 or min_spacing < 0 or max_points < 1:
        raise ValueError("Invalid BPO budget, spacing, or point limit")
    if budget == 0:
        return []
    if len({x["step"] for x in scores}) != len(scores):
        raise ValueError("Duplicate backbone checkpoint")
    for score in scores:
        if not math.isfinite(score["entropy_lower_bound_nats"]) or score["entropy_lower_bound_nats"] < 0:
            raise ValueError("Invalid entropy score")
        if type(score["token_position"]) is not int or score["token_position"] < 0:
            raise ValueError("Missing or invalid completion-token position")
    ranked = sorted(scores, key=lambda x: (-x["entropy_lower_bound_nats"], x["step"]))
    selected = []
    for score in ranked:
        if all(abs(score["token_position"] - other["token_position"]) >= min_spacing
               for other in selected):
            selected.append(score)
            if len(selected) >= min(budget, max_points):
                break
    if not selected:
        raise ValueError("No eligible probability-scored branch points")
    return [dict(selected[i % len(selected)], sibling_index=2 + i // len(selected))
            for i in range(budget)]


def entropy_record(token: dict) -> dict:
    entropy = asdict(token_entropy(token))
    entropy["entropy_lower_bound_nats"] = entropy.pop("lower_bound_nats")
    return entropy
