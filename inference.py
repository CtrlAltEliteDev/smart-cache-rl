"""
Inference script for smart_cache_rl.

Runs all three tasks (easy → medium → hard) in-process and emits
structured [START] / [STEP] / [END] logs for each.

Required env vars:
    API_BASE_URL   LiteLLM proxy endpoint (used when POLICY_MODE=llm)
    API_KEY        LiteLLM proxy API key (used when POLICY_MODE=llm)
    MODEL_NAME     Model identifier (used only when POLICY_MODE=llm)

Optional env vars:
    POLICY_MODE    one of: lru | lfu | dqn | llm  (default: llm)
    TEMPERATURE    LLM sampling temperature (default: 0.2)
    MAX_TOKENS     LLM max tokens per call (default: 80)
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from openai import OpenAI

from agent import DQNCacheAgent
from models import SmartCacheRlAction, SmartCacheRlObservation
from policy import choose_lfu_eviction, choose_lru_eviction
from tasks import TASKS, Task, strict_open_unit

# Import the environment directly for in-process execution
try:
    from server.smart_cache_rl_environment import SmartCacheRlEnvironment
except ImportError:
    from smart_cache_rl_environment import SmartCacheRlEnvironment


API_BASE_URL = os.getenv("API_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("API_KEY", "").strip()
BENCHMARK = os.getenv("BENCHMARK", "smart_cache_rl")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "80"))
POLICY_MODE = os.getenv("POLICY_MODE", "llm").strip().lower()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


def _build_prompt(obs: SmartCacheRlObservation, capacity: int) -> str:
    return (
        "You are choosing a cache eviction slot.\n"
        f"Return exactly one token: NONE or an integer index between 0 and {capacity - 1}.\n"
        f"incoming_item_id={obs.incoming_item_id}\n"
        f"incoming_size={obs.incoming_size:.4f}\n"
        f"incoming_cost={obs.incoming_cost:.4f}\n"
        f"incoming_popularity={obs.incoming_popularity:.6f}\n"
        f"cache_fill_ratio={obs.cache_fill_ratio:.4f}\n"
        f"cache_item_ids={obs.cache_item_ids}\n"
        f"cache_recency={obs.cache_recency}\n"
        f"cache_frequency={obs.cache_frequency}\n"
        f"cache_size={obs.cache_size}\n"
        f"cache_cost={obs.cache_cost}\n"
    )


def _parse_action(text: str, capacity: int) -> Optional[int]:
    raw = (text or "").strip().upper()
    if raw == "NONE":
        return None
    m = re.search(r"-?\d+", raw)
    if not m:
        return None
    idx = int(m.group(0))
    return max(0, min(capacity - 1, idx))


def choose_action_with_llm(
    client: OpenAI, obs: SmartCacheRlObservation, capacity: int
) -> Optional[int]:
    if obs.cache_fill_ratio < 1.0:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are an RL cache policy helper. Output only NONE or a single integer index.",
                },
                {"role": "user", "content": _build_prompt(obs, capacity)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = _parse_action(text, capacity)
        if parsed is None and obs.cache_fill_ratio >= 1.0:
            return choose_lru_eviction(obs)
        return parsed
    except Exception:
        return choose_lru_eviction(obs)


def run_task(task: Task, client: Optional[OpenAI]) -> float:
    """Run one complete episode for the given task. Returns score in (0, 1), never 0.0 or 1.0."""
    capacity = int(task.env_config["CACHE_CAPACITY"])
    max_steps = int(task.env_config["MAX_STEPS"])
    feature_dim = 4 * capacity + 4

    env = SmartCacheRlEnvironment(config=task.env_config)
    dqn_agent = DQNCacheAgent(capacity=capacity, feature_dim=feature_dim, seed=42)

    rewards: List[float] = []
    steps_taken = 0
    prev_obs: Optional[SmartCacheRlObservation] = None

    log_start(task=task.name, env=BENCHMARK, model=MODEL_NAME)
    success = False
    score = strict_open_unit(0.0)
    try:
        obs = env.reset()
        for step in range(1, max_steps + 1):
            if obs.done:
                break

            if POLICY_MODE == "lru":
                evict_index = choose_lru_eviction(obs)
            elif POLICY_MODE == "lfu":
                evict_index = choose_lfu_eviction(obs)
            elif POLICY_MODE == "llm":
                evict_index = choose_action_with_llm(client, obs, capacity)  # type: ignore[arg-type]
            else:
                evict_index = dqn_agent.act(obs, training=True)

            action_str = "NONE" if evict_index is None else str(evict_index)
            prev_obs = obs
            obs = env.step(SmartCacheRlAction(evict_index=evict_index))
            reward = float(obs.reward or 0.0)
            done = bool(obs.done)

            if POLICY_MODE == "dqn":
                dqn_agent.observe(prev_obs, evict_index, reward, obs, done)
                dqn_agent.train_step(batch_size=32)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            if done:
                break

        hit_rate = float((obs.metadata or {}).get("hit_rate", 0.0))
        score = strict_open_unit(task.grader(hit_rate))
        success = hit_rate >= task.success_threshold
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    if POLICY_MODE == "llm" and not API_BASE_URL:
        raise RuntimeError("API_BASE_URL is required when POLICY_MODE=llm")
    if POLICY_MODE == "llm" and not API_KEY:
        raise RuntimeError("API_KEY is required when POLICY_MODE=llm")

    client: Optional[OpenAI] = None
    if POLICY_MODE == "llm":
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task in TASKS:
        run_task(task, client)


if __name__ == "__main__":
    main()
