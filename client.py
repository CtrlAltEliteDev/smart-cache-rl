# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Cache RL Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SmartCacheRlAction, SmartCacheRlObservation


class SmartCacheRlEnv(
    EnvClient[SmartCacheRlAction, SmartCacheRlObservation, State]
):
    """
    Client for the Smart Cache RL Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> with SmartCacheRlEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     action = SmartCacheRlAction(evict_index=0)
        ...     result = client.step(action)

    Example with Docker:
        >>> client = SmartCacheRlEnv.from_docker_image("smart_cache_rl-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SmartCacheRlAction(evict_index=0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SmartCacheRlAction) -> Dict:
        """
        Convert SmartCacheRlAction to JSON payload for step message.

        Args:
            action: SmartCacheRlAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "evict_index": action.evict_index,
            "use_agent": action.use_agent,
            "agent_mode": action.agent_mode,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SmartCacheRlObservation]:
        """
        Parse server response into StepResult[SmartCacheRlObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SmartCacheRlObservation
        """
        obs_data = payload.get("observation", {})
        observation = SmartCacheRlObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
