"""Baseline eviction policies: LRU and LFU."""

from __future__ import annotations

from typing import Optional

try:
    from .models import SmartCacheRlObservation
except ImportError:
    from models import SmartCacheRlObservation


def choose_lru_eviction(observation: SmartCacheRlObservation) -> Optional[int]:
    """Pick slot with smallest recency value."""
    if observation.cache_fill_ratio < 1.0:
        return None
    recency = observation.cache_recency
    min_idx = 0
    min_val = recency[0]
    for i, v in enumerate(recency):
        if v < min_val:
            min_val = v
            min_idx = i
    return min_idx


def choose_lfu_eviction(observation: SmartCacheRlObservation) -> Optional[int]:
    """Pick slot with smallest frequency value."""
    if observation.cache_fill_ratio < 1.0:
        return None
    freq = observation.cache_frequency
    min_idx = 0
    min_val = freq[0]
    for i, v in enumerate(freq):
        if v < min_val:
            min_val = v
            min_idx = i
    return min_idx
