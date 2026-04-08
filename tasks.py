"""Task definitions for Smart Cache RL with deterministic graders.

Three tasks ranging from easy to hard, each with a programmatic grader
that returns a score strictly in (0.0, 1.0).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, NamedTuple

# Open interval (0, 1) — hackathon / evaluator checks reject 0.0 and 1.0 exactly.
EPS_OPEN: float = 1e-5


def strict_open_unit(x: float) -> float:
    """Map a scalar into the open interval (0, 1); never returns 0.0 or 1.0."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return EPS_OPEN
    if not math.isfinite(v):
        return EPS_OPEN
    return min(1.0 - EPS_OPEN, max(EPS_OPEN, v))


class Task(NamedTuple):
    name: str
    description: str
    difficulty: str          # easy | medium | hard
    env_config: Dict[str, Any]
    grader: Callable[[float], float]
    success_threshold: float  # minimum hit_rate for success=True


def _easy_grader(hit_rate: float) -> float:
    """4-slot cache, 16-item Zipf traffic. LRU ≈90%.
    Score approaches 0 at very low hit rate and approaches 1 at 80%+ hit rate."""
    return strict_open_unit(hit_rate / 0.80)


def _medium_grader(hit_rate: float) -> float:
    """16-slot cache, 128-item Zipf traffic. LRU ≈97%.
    Score approaches 0 at 60% hit rate and approaches 1 at 95%+ hit rate."""
    return strict_open_unit((hit_rate - 0.60) / 0.35)


def _hard_grader(hit_rate: float) -> float:
    """8-slot cache, adversarial Zipf with midpoint shift. LRU ≈97% pre-shift, drops after.
    Score approaches 0 at 50% hit rate and approaches 1 at 90%+ hit rate."""
    return strict_open_unit((hit_rate - 0.50) / 0.40)


TASKS: List[Task] = [
    Task(
        name="cache-warmup",
        description=(
            "Warm an empty 4-slot cache under Zipf-distributed traffic from a 16-item catalog. "
            "A small number of hot items dominate requests; once the cache warms up, any policy "
            "that retains recently-seen items achieves a high hit rate. Episode length: 100 steps."
        ),
        difficulty="easy",
        env_config={
            "CACHE_CAPACITY": 4,
            "CATALOG_SIZE": 16,
            "MAX_STEPS": 100,
            "REQUEST_SAMPLING_MODE": "popularity",
            "USE_WIKIPEDIA_DATA": False,
            "ADVERSARIAL": False,
        },
        grader=_easy_grader,
        success_threshold=0.40,
    ),
    Task(
        name="cache-eviction",
        description=(
            "Manage a 16-slot cache under Zipf-distributed popularity (α=1.25) over a 128-item "
            "catalog. A small fraction of items drives most traffic; the agent must identify and "
            "retain high-value items while evicting low-value ones. Episode length: 500 steps."
        ),
        difficulty="medium",
        env_config={
            "CACHE_CAPACITY": 16,
            "CATALOG_SIZE": 128,
            "MAX_STEPS": 500,
            "REQUEST_SAMPLING_MODE": "popularity",
            "USE_WIKIPEDIA_DATA": False,
            "ADVERSARIAL": False,
        },
        grader=_medium_grader,
        success_threshold=0.50,
    ),
    Task(
        name="cache-adversarial",
        description=(
            "Manage an 8-slot cache where the Zipf popularity distribution shifts sharply at the "
            "episode midpoint: items that dominated the first 300 steps become rare, and previously "
            "rare items surge in popularity. The agent must rapidly evict stale items and admit "
            "newly popular ones. Episode length: 600 steps."
        ),
        difficulty="hard",
        env_config={
            "CACHE_CAPACITY": 8,
            "CATALOG_SIZE": 128,
            "MAX_STEPS": 600,
            "REQUEST_SAMPLING_MODE": "popularity",
            "USE_WIKIPEDIA_DATA": False,
            "ADVERSARIAL": True,
        },
        grader=_hard_grader,
        success_threshold=0.35,
    ),
]

TASKS_BY_NAME: Dict[str, Task] = {t.name: t for t in TASKS}
