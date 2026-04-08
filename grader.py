"""Episode grader for live cache RL evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

_EPS = 1e-6


def _strict_unit(x: float) -> float:
    return min(1.0 - _EPS, max(_EPS, float(x)))


@dataclass
class Grader:
    """Tracks request/hit/miss/training metrics during an episode."""

    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    cumulative_reward: float = 0.0
    penalties: int = 0
    last_reward: float = 0.0
    last_hit: bool = False
    last_training_loss: float = 0.0
    last_epsilon: float = 0.0
    train_steps: int = 0
    replay_size: int = 0
    grade: float = 0.0
    history: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.total_requests = 0
        self.hits = 0
        self.misses = 0
        self.cumulative_reward = 0.0
        self.penalties = 0
        self.last_reward = 0.0
        self.last_hit = False
        self.last_training_loss = 0.0
        self.last_epsilon = 0.0
        self.train_steps = 0
        self.replay_size = 0
        self.grade = 0.0
        self.history.clear()

    def update(
        self,
        *,
        is_hit: bool,
        reward: float,
        training_loss: float,
        epsilon: float,
        train_steps: int,
        replay_size: int,
    ) -> None:
        self.total_requests += 1
        if is_hit:
            self.hits += 1
        else:
            self.misses += 1
        if reward < 0.0:
            self.penalties += 1
        self.last_reward = float(reward)
        self.last_hit = bool(is_hit)
        self.cumulative_reward += float(reward)
        self.last_training_loss = float(training_loss)
        self.last_epsilon = float(epsilon)
        self.train_steps = int(train_steps)
        self.replay_size = int(replay_size)
        self.history.append(float(reward))
        if len(self.history) > 200:
            self.history = self.history[-200:]
        self.grade = self._compute_grade()

    def _compute_grade(self) -> float:
        if self.total_requests <= 0:
            return 0.0
        hit_rate = float(self.hits) / float(self.total_requests)
        avg_reward = self.cumulative_reward / float(self.total_requests)
        reward_term = max(0.0, min(1.0, (avg_reward + 30.0) / 60.0))
        return 0.7 * hit_rate + 0.3 * reward_term

    def to_dict(self) -> Dict[str, float | int | bool]:
        hit_rate = float(self.hits) / max(1, self.total_requests)
        return {
            "total_requests": self.total_requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cumulative_reward": self.cumulative_reward,
            "penalties": self.penalties,
            "last_reward": self.last_reward,
            "last_hit": self.last_hit,
            "training_loss": self.last_training_loss,
            "epsilon": self.last_epsilon,
            "train_steps": self.train_steps,
            "replay_size": self.replay_size,
            "grade": self.grade,
        }


def score_episode(task_name: str, hit_rate: float) -> float:
    """Return a deterministic score in [0.0, 1.0] for a completed episode.

    Args:
        task_name: One of 'cache-warmup', 'cache-eviction', 'cache-adversarial'.
        hit_rate:  Fraction of requests that were cache hits during the episode.

    Returns:
        Normalized score strictly in (0.0, 1.0).
    """
    try:
        from tasks import TASKS_BY_NAME
        task = TASKS_BY_NAME[task_name]
        return task.grader(hit_rate)
    except KeyError:
        # Unknown task – fall back to raw hit rate
        return _strict_unit(hit_rate)
