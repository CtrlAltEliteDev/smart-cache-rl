# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Cache RL Environment Implementation."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, List, Optional
from uuid import uuid4

import numpy as np

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SmartCacheRlAction, SmartCacheRlObservation
    from ..agent import DQNCacheAgent
except ImportError:
    from models import SmartCacheRlAction, SmartCacheRlObservation
    from agent import DQNCacheAgent

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None


class SmartCacheRlEnvironment(Environment):
    """Cache simulator with RL-compatible eviction control."""

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    @dataclass
    class _CacheItem:
        item_id: int
        last_access_step: int
        frequency: int

    def __init__(self):
        self.capacity = 16
        self.catalog_size = 128
        self.max_steps = 1200
        self.default_seed = 42

        self._rng = np.random.default_rng(self.default_seed)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._current_seed = self.default_seed

        self._item_sizes = self._rng.integers(1, 16, size=self.catalog_size).astype(float)
        self._item_costs = self._rng.integers(1, 32, size=self.catalog_size).astype(float)
        raw_pop = self._rng.zipf(1.25, size=self.catalog_size).astype(float)
        self._popularity = raw_pop / raw_pop.sum()

        self._cache: List[SmartCacheRlEnvironment._CacheItem] = []
        self._slot_by_item: Dict[int, int] = {}
        self._next_request: Optional[int] = None
        self._last_hit = False
        self._last_reward = 0.0
        self._hit_count = 0
        self._miss_count = 0
        self._redis_enabled = os.getenv("USE_REDIS", "false").lower() in ("1", "true", "yes")
        self._redis_url = os.getenv("REDIS_URL", "").strip()
        self._redis_client = None
        self._redis_status = "disabled"
        self._redis_episode_key = ""
        self._init_redis()
        self._agent_mode = os.getenv("AGENT_MODE", "manual").strip().lower()
        if self._agent_mode not in ("manual", "lru", "lfu", "dqn"):
            self._agent_mode = "manual"
        self._dqn_agent = DQNCacheAgent(capacity=self.capacity, feature_dim=68, seed=42)
        self._last_applied_evict_slot = -1
        self._last_action_source = "manual"
        self._last_agent_suggested_slot = -1

    def reset(self) -> SmartCacheRlObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._cache = []
        self._slot_by_item = {}
        self._last_hit = False
        self._last_reward = 0.0
        self._hit_count = 0
        self._miss_count = 0

        self._current_seed = self.default_seed
        self._rng = np.random.default_rng(self._current_seed)
        self._next_request = int(self._rng.choice(self.catalog_size, p=self._popularity))
        self._last_applied_evict_slot = -1
        self._last_action_source = "manual"
        self._last_agent_suggested_slot = -1
        self._redis_episode_key = f"smart_cache_rl:episode:{self._state.episode_id}"
        self._write_redis_metrics(event="reset")
        return self._build_observation(done=False, reward=0.0)

    def step(self, action: SmartCacheRlAction) -> SmartCacheRlObservation:  # type: ignore[override]
        if self._next_request is None:
            self._next_request = int(self._rng.choice(self.catalog_size, p=self._popularity))

        self._state.step_count += 1
        request_id = int(self._next_request)
        self._last_hit = request_id in self._slot_by_item
        invalid_action_penalty = 0.0

        if self._last_hit:
            slot = self._slot_by_item[request_id]
            entry = self._cache[slot]
            entry.frequency += 1
            entry.last_access_step = self._state.step_count
            reward = float(self._item_costs[request_id])
            self._hit_count += 1
        else:
            self._miss_count += 1
            reward = -float(self._item_costs[request_id])
            if len(self._cache) < self.capacity:
                self._insert_new_item(request_id)
                self._last_applied_evict_slot = -1
                self._last_action_source = "none:not_full"
            else:
                effective_mode = self._resolve_agent_mode(action)
                self._last_agent_suggested_slot = self._suggest_slot(effective_mode)
                auto_mode = action.use_agent or (action.evict_index is None and effective_mode != "manual")
                if auto_mode:
                    evict_index = self._last_agent_suggested_slot
                    self._last_action_source = "agent:auto"
                elif action.evict_index is None:
                    evict_index = self._evict_lru_index()
                    invalid_action_penalty = -0.25
                    self._last_action_source = "agent:fallback"
                else:
                    evict_index = max(0, min(self.capacity - 1, int(action.evict_index)))
                    if evict_index != action.evict_index:
                        invalid_action_penalty = -0.25
                    self._last_action_source = "manual"
                self._last_applied_evict_slot = evict_index
                self._evict_and_insert(evict_index, request_id)

        reward += invalid_action_penalty
        self._last_reward = reward
        done = self._state.step_count >= self.max_steps
        self._next_request = int(self._rng.choice(self.catalog_size, p=self._popularity)) if not done else None
        self._write_redis_metrics(event="step")
        return self._build_observation(done=done, reward=reward)

    @property
    def state(self) -> State:
        self._state.hit_count = self._hit_count  # type: ignore[attr-defined]
        self._state.miss_count = self._miss_count  # type: ignore[attr-defined]
        self._state.hit_rate = (  # type: ignore[attr-defined]
            float(self._hit_count) / max(1, self._hit_count + self._miss_count)
        )
        return self._state

    def _insert_new_item(self, item_id: int) -> None:
        self._cache.append(
            self._CacheItem(
                item_id=item_id,
                last_access_step=self._state.step_count,
                frequency=1,
            )
        )
        self._slot_by_item[item_id] = len(self._cache) - 1

    def _evict_and_insert(self, slot_index: int, new_item_id: int) -> None:
        evicted_id = self._cache[slot_index].item_id
        del self._slot_by_item[evicted_id]
        self._cache[slot_index] = self._CacheItem(
            item_id=new_item_id,
            last_access_step=self._state.step_count,
            frequency=1,
        )
        self._slot_by_item[new_item_id] = slot_index

    def _evict_lru_index(self) -> int:
        min_step = min(x.last_access_step for x in self._cache)
        for i, x in enumerate(self._cache):
            if x.last_access_step == min_step:
                return i
        return 0

    def _evict_lfu_index(self) -> int:
        min_freq = min(x.frequency for x in self._cache)
        for i, x in enumerate(self._cache):
            if x.frequency == min_freq:
                return i
        return 0

    def _resolve_agent_mode(self, action: SmartCacheRlAction) -> str:
        mode = (action.agent_mode or self._agent_mode or "manual").strip().lower()
        if mode not in ("manual", "lru", "lfu", "dqn"):
            return "manual"
        return mode

    def _suggest_slot(self, mode: str) -> int:
        if len(self._cache) < self.capacity:
            return -1
        if mode == "lru":
            return self._evict_lru_index()
        if mode == "lfu":
            return self._evict_lfu_index()
        if mode == "dqn":
            x = self._build_dqn_feature_vector()
            q = self._dqn_agent.weights @ x
            idx = int(np.argmax(q))
            return int(max(0, min(self.capacity - 1, idx)))
        return self._evict_lru_index()

    def _build_dqn_feature_vector(self) -> np.ndarray:
        recency: List[float] = []
        frequency: List[float] = []
        size: List[float] = []
        cost: List[float] = []
        max_freq = max([x.frequency for x in self._cache], default=1)
        for slot in range(self.capacity):
            if slot < len(self._cache):
                entry = self._cache[slot]
                recency.append(
                    1.0 / (1.0 + float(max(0, self._state.step_count - entry.last_access_step)))
                )
                frequency.append(float(entry.frequency) / float(max_freq))
                size.append(float(self._item_sizes[entry.item_id]) / float(self._item_sizes.max()))
                cost.append(float(self._item_costs[entry.item_id]) / float(self._item_costs.max()))
            else:
                recency.append(0.0)
                frequency.append(0.0)
                size.append(0.0)
                cost.append(0.0)

        incoming_id = int(self._next_request) if self._next_request is not None else -1
        incoming_size = (
            float(self._item_sizes[incoming_id]) / float(self._item_sizes.max())
            if incoming_id >= 0
            else 0.0
        )
        incoming_cost = (
            float(self._item_costs[incoming_id]) / float(self._item_costs.max())
            if incoming_id >= 0
            else 0.0
        )
        incoming_pop = float(self._popularity[incoming_id]) if incoming_id >= 0 else 0.0
        fill_ratio = float(len(self._cache)) / float(self.capacity)
        vec = recency + frequency + size + cost + [incoming_size, incoming_cost, incoming_pop, fill_ratio]
        return np.asarray(vec, dtype=np.float32)

    def _build_observation(self, done: bool, reward: float) -> SmartCacheRlObservation:
        recency: List[float] = []
        frequency: List[float] = []
        size: List[float] = []
        cost: List[float] = []
        occupied: List[int] = []
        ids: List[int] = []

        max_freq = max([x.frequency for x in self._cache], default=1)
        for slot in range(self.capacity):
            if slot < len(self._cache):
                entry = self._cache[slot]
                ids.append(entry.item_id)
                occupied.append(1)
                recency.append(
                    1.0
                    / (1.0 + float(max(0, self._state.step_count - entry.last_access_step)))
                )
                frequency.append(float(entry.frequency) / float(max_freq))
                size.append(float(self._item_sizes[entry.item_id]) / float(self._item_sizes.max()))
                cost.append(float(self._item_costs[entry.item_id]) / float(self._item_costs.max()))
            else:
                ids.append(-1)
                occupied.append(0)
                recency.append(0.0)
                frequency.append(0.0)
                size.append(0.0)
                cost.append(0.0)

        incoming_id = int(self._next_request) if self._next_request is not None else -1
        incoming_size = (
            float(self._item_sizes[incoming_id]) / float(self._item_sizes.max())
            if incoming_id >= 0
            else 0.0
        )
        incoming_cost = (
            float(self._item_costs[incoming_id]) / float(self._item_costs.max())
            if incoming_id >= 0
            else 0.0
        )
        incoming_pop = float(self._popularity[incoming_id]) if incoming_id >= 0 else 0.0
        total = self._hit_count + self._miss_count
        hit_rate = float(self._hit_count) / max(1, total)
        lru_slot = self._evict_lru_index() if len(self._cache) == self.capacity else -1
        if len(self._cache) == self.capacity:
            lfu_slot = self._evict_lfu_index()
            dqn_slot = self._suggest_slot("dqn")
        else:
            lfu_slot = -1
            dqn_slot = -1
        snapshot: List[str] = []
        for slot in range(self.capacity):
            if slot < len(self._cache):
                entry = self._cache[slot]
                age = self._state.step_count - entry.last_access_step
                snapshot.append(
                    f"slot={slot} item={entry.item_id} freq={entry.frequency} age={age} "
                    f"size={self._item_sizes[entry.item_id]:.0f} cost={self._item_costs[entry.item_id]:.0f}"
                )
            else:
                snapshot.append(f"slot={slot} empty")

        event = "HIT" if self._last_hit else "MISS"
        ui_summary = (
            f"step={self._state.step_count} {event} reward={reward:.2f} "
            f"incoming={incoming_id} fill={sum(occupied)}/{self.capacity}"
        )
        ui_hit_rate = f"hits={self._hit_count} misses={self._miss_count} hit_rate={hit_rate:.3f}"
        redis_latest = self._read_redis_latest()
        redis_line = (
            f"redis={self._redis_status} latest_hit_rate={redis_latest.get('hit_rate', 'n/a')} "
            f"latest_step={redis_latest.get('step_count', 'n/a')}"
        )

        return SmartCacheRlObservation(
            cache_item_ids=ids,
            cache_recency=recency,
            cache_frequency=frequency,
            cache_size=size,
            cache_cost=cost,
            cache_occupied=occupied,
            incoming_item_id=incoming_id,
            incoming_size=incoming_size,
            incoming_cost=incoming_cost,
            incoming_popularity=incoming_pop,
            is_hit=self._last_hit,
            cache_fill_ratio=float(sum(occupied)) / float(self.capacity),
            ui_summary=ui_summary,
            ui_hit_rate=ui_hit_rate,
            ui_suggested_lru_slot=lru_slot,
            ui_suggested_lfu_slot=lfu_slot,
            ui_suggested_dqn_slot=dqn_slot,
            ui_agent_mode=self._agent_mode,
            ui_action_source=self._last_action_source,
            ui_applied_evict_slot=self._last_applied_evict_slot,
            ui_cache_snapshot=snapshot,
            ui_redis_status_line=redis_line,
            done=done,
            reward=reward,
            metadata={
                "step": self._state.step_count,
                "hits": self._hit_count,
                "misses": self._miss_count,
                "hit_rate": hit_rate,
                "redis_enabled": self._redis_enabled,
                "redis_status": self._redis_status,
                "redis_episode_key": self._redis_episode_key,
                "agent_mode": self._agent_mode,
                "agent_suggested_action": self._last_agent_suggested_slot,
                "action_source": self._last_action_source,
                "applied_evict_slot": self._last_applied_evict_slot,
                "redis_latest": redis_latest,
            },
        )

    def _init_redis(self) -> None:
        if not self._redis_enabled:
            self._redis_status = "disabled"
            return
        if redis is None:
            self._redis_status = "redis_package_missing"
            return
        if not self._redis_url:
            self._redis_status = "redis_url_missing"
            return
        try:
            self._redis_client = redis.from_url(
                self._redis_url,
                decode_responses=True,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
            )
            self._redis_client.ping()
            self._redis_status = "connected"
        except Exception:
            self._redis_client = None
            self._redis_status = "connection_failed_fallback_memory"

    def _write_redis_metrics(self, event: str) -> None:
        if self._redis_client is None:
            return
        total = self._hit_count + self._miss_count
        hit_rate = float(self._hit_count) / max(1, total)
        payload = {
            "event": event,
            "episode_id": str(self._state.episode_id or ""),
            "step_count": str(self._state.step_count),
            "hits": str(self._hit_count),
            "misses": str(self._miss_count),
            "hit_rate": f"{hit_rate:.6f}",
            "cache_fill_ratio": f"{len(self._cache) / float(self.capacity):.6f}",
            "last_reward": f"{self._last_reward:.6f}",
        }
        try:
            if self._redis_episode_key:
                self._redis_client.hset(self._redis_episode_key, mapping=payload)
                self._redis_client.expire(self._redis_episode_key, 60 * 60 * 24)
            self._redis_client.hset("smart_cache_rl:latest", mapping=payload)
            self._redis_status = "connected"
        except Exception:
            self._redis_status = "write_failed_fallback_memory"

    def _read_redis_latest(self) -> Dict[str, str]:
        if self._redis_client is None:
            return {}
        try:
            data = self._redis_client.hgetall("smart_cache_rl:latest")
            self._redis_status = "connected"
            return data if isinstance(data, dict) else {}
        except Exception:
            self._redis_status = "read_failed_fallback_memory"
            return {}
