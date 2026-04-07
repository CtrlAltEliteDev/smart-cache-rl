"""
Inference script for smart_cache_rl.

Required env vars:
    API_BASE_URL
    MODEL_NAME
    HF_TOKEN
    LOCAL_IMAGE_NAME (optional if ENV_BASE_URL is used)
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from openai import OpenAI

from agent import DQNCacheAgent
from client import SmartCacheRlEnv
from models import SmartCacheRlAction
from policy import choose_lfu_eviction, choose_lru_eviction


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "").strip()
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000").strip()

TASK_NAME = os.getenv("TASK_NAME", "smart-cache-eviction")
BENCHMARK = os.getenv("BENCHMARK", "smart_cache_rl")
MAX_STEPS = int(os.getenv("MAX_STEPS", "200"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "80"))
POLICY_MODE = os.getenv("POLICY_MODE", "dqn").strip().lower()


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _build_prompt(obs) -> str:
    return (
        "You are choosing a cache eviction slot.\n"
        "Return exactly one token: NONE or an integer index between 0 and 15.\n"
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


def _parse_action(text: str) -> Optional[int]:
    raw = (text or "").strip().upper()
    if raw == "NONE":
        return None
    m = re.search(r"-?\d+", raw)
    if not m:
        return None
    idx = int(m.group(0))
    if idx < 0:
        return 0
    if idx > 15:
        return 15
    return idx


def choose_action_with_llm(client: OpenAI, obs) -> Optional[int]:
    if obs.cache_fill_ratio < 1.0:
        return None
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an RL cache policy helper. "
                        "Output only NONE or a single integer index."
                    ),
                },
                {"role": "user", "content": _build_prompt(obs)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()
        parsed = _parse_action(text)
        if parsed is None and obs.cache_fill_ratio >= 1.0:
            return choose_lru_eviction(obs)
        return parsed
    except Exception:
        return choose_lru_eviction(obs)


def main() -> None:
    if POLICY_MODE == "llm" and not API_KEY:
        raise RuntimeError("HF_TOKEN is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY) if POLICY_MODE == "llm" else None
    env = (
        SmartCacheRlEnv.from_docker_image(LOCAL_IMAGE_NAME)
        if LOCAL_IMAGE_NAME
        else SmartCacheRlEnv(base_url=ENV_BASE_URL)
    )
    dqn_agent = DQNCacheAgent(capacity=16, feature_dim=68, seed=42)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
    try:
        result = env.reset()
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break
            obs = result.observation
            if POLICY_MODE == "lru":
                evict_index = choose_lru_eviction(obs)
            elif POLICY_MODE == "lfu":
                evict_index = choose_lfu_eviction(obs)
            elif POLICY_MODE == "llm":
                evict_index = choose_action_with_llm(client, obs)  # type: ignore[arg-type]
            else:
                evict_index = dqn_agent.act(obs, training=True)
            action_str = "NONE" if evict_index is None else str(evict_index)

            result = env.step(SmartCacheRlAction(evict_index=evict_index))
            reward = float(result.reward or 0.0)
            done = bool(result.done)
            next_obs = result.observation

            if POLICY_MODE == "dqn":
                dqn_agent.observe(obs, evict_index, reward, next_obs, done)
                dqn_agent.train_step(batch_size=32)

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            if done:
                break

        final_obs = result.observation
        hit_rate = float((final_obs.metadata or {}).get("hit_rate", 0.0))
        score = max(0.0, min(1.0, hit_rate))
        success = score >= 0.5
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
