"""Lightweight DQN-style agent for cache eviction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from .models import SmartCacheRlObservation
    from .policy import choose_lru_eviction
except ImportError:
    from models import SmartCacheRlObservation
    from policy import choose_lru_eviction


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class DQNCacheAgent:
    """Small linear Q approximator for fast training/inference."""

    def __init__(self, capacity: int, feature_dim: int, seed: int = 42):
        self.capacity = capacity
        self.feature_dim = feature_dim
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((capacity, feature_dim), dtype=np.float32)
        self.replay: List[Transition] = []
        self.gamma = 0.95
        self.lr = 0.02
        self.epsilon = 0.15
        self.max_replay = 5000

    def vectorize(self, obs: SmartCacheRlObservation) -> np.ndarray:
        return np.asarray(
            obs.cache_recency
            + obs.cache_frequency
            + obs.cache_size
            + obs.cache_cost
            + [obs.incoming_size, obs.incoming_cost, obs.incoming_popularity, obs.cache_fill_ratio],
            dtype=np.float32,
        )

    def act(self, obs: SmartCacheRlObservation, training: bool = False) -> Optional[int]:
        if obs.cache_fill_ratio < 1.0:
            return None
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.capacity))
        x = self.vectorize(obs)
        q = self.weights @ x
        best = int(np.argmax(q))
        if np.allclose(q, q[0]):
            fallback = choose_lru_eviction(obs)
            return 0 if fallback is None else fallback
        return best

    def observe(
        self,
        obs: SmartCacheRlObservation,
        action: Optional[int],
        reward: float,
        next_obs: SmartCacheRlObservation,
        done: bool,
    ) -> None:
        if action is None:
            return
        t = Transition(
            state=self.vectorize(obs),
            action=action,
            reward=reward,
            next_state=self.vectorize(next_obs),
            done=done,
        )
        self.replay.append(t)
        if len(self.replay) > self.max_replay:
            self.replay.pop(0)

    def train_step(self, batch_size: int = 32) -> float:
        if not self.replay:
            return 0.0
        idxs = self.rng.integers(0, len(self.replay), size=min(batch_size, len(self.replay)))
        loss_sum = 0.0
        for i in idxs:
            tr = self.replay[int(i)]
            q_pred = float(self.weights[tr.action] @ tr.state)
            q_next = self.weights @ tr.next_state
            target = tr.reward + (0.0 if tr.done else self.gamma * float(np.max(q_next)))
            td_error = target - q_pred
            self.weights[tr.action] += self.lr * td_error * tr.state
            loss_sum += td_error * td_error
        return loss_sum / float(len(idxs))
