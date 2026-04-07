# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the Smart Cache RL environment."""

from typing import List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SmartCacheRlAction(Action):
    """
    Cache eviction action.

    The agent only needs to choose `evict_index` when the cache is full and
    the incoming item is not already cached. In all other cases, this can be
    omitted.
    """

    evict_index: Optional[int] = Field(
        default=None,
        description="Cache slot index to evict when replacement is required.",
    )
    use_agent: bool = Field(
        default=False,
        description="If true, let environment choose eviction using configured agent mode.",
    )
    agent_mode: Optional[str] = Field(
        default=None,
        description="Optional per-step override: manual|lru|lfu|dqn.",
    )


class SmartCacheRlObservation(Observation):
    """Observation containing cache state and current request features."""

    cache_item_ids: List[int] = Field(default_factory=list, description="Item id per slot.")
    cache_recency: List[float] = Field(
        default_factory=list,
        description="Normalized recency per slot (higher = more recently used).",
    )
    cache_frequency: List[float] = Field(
        default_factory=list,
        description="Normalized access frequency per slot.",
    )
    cache_size: List[float] = Field(
        default_factory=list,
        description="Normalized size per slot.",
    )
    cache_cost: List[float] = Field(
        default_factory=list,
        description="Normalized fetch cost per slot.",
    )
    cache_occupied: List[int] = Field(
        default_factory=list,
        description="1 if slot occupied else 0.",
    )
    incoming_item_id: int = Field(default=-1, description="Requested item id.")
    incoming_size: float = Field(default=0.0, description="Normalized size of requested item.")
    incoming_cost: float = Field(default=0.0, description="Normalized fetch cost of requested item.")
    incoming_popularity: float = Field(
        default=0.0, description="Normalized request probability of requested item."
    )
    is_hit: bool = Field(default=False, description="True when request is already in cache.")
    cache_fill_ratio: float = Field(default=0.0, description="Current cache utilization.")
    ui_summary: str = Field(
        default="",
        description="Human-readable summary for the web UI.",
    )
    ui_hit_rate: str = Field(
        default="",
        description="Formatted hit-rate summary string for the web UI.",
    )
    ui_suggested_lru_slot: int = Field(
        default=-1,
        description="Suggested eviction slot by LRU heuristic.",
    )
    ui_suggested_lfu_slot: int = Field(
        default=-1,
        description="Suggested eviction slot by LFU heuristic.",
    )
    ui_suggested_dqn_slot: int = Field(
        default=-1,
        description="Suggested eviction slot by DQN policy.",
    )
    ui_agent_mode: str = Field(
        default="manual",
        description="Current agent mode used by the environment.",
    )
    ui_action_source: str = Field(
        default="manual",
        description="Source of applied action: manual|agent:auto|agent:fallback.",
    )
    ui_applied_evict_slot: int = Field(
        default=-1,
        description="Eviction slot applied on the last step, else -1.",
    )
    ui_cache_snapshot: List[str] = Field(
        default_factory=list,
        description="Readable one-line snapshot per cache slot.",
    )
    ui_redis_status_line: str = Field(
        default="",
        description="Readable Redis status/latest metrics line for UI.",
    )
