---
title: Smart Cache RL
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - cache-eviction
---

# Smart Cache RL Environment

Smart Cache RL is an OpenEnv environment for cache-eviction experiments.
It simulates a fixed-size cache and compares manual eviction decisions with built-in policies (`lru`, `lfu`, `dqn`) under skewed request traffic.

## What This Project Is For

Use this project to:
- Prototype cache eviction behavior quickly with a web UI.
- Compare heuristic policies (LRU/LFU) against a simple RL policy (DQN-style linear Q approximator).
- Inspect per-step reward, hit-rate, and cache state transitions.
- Optionally stream metrics to Redis.

## Core Behavior

- Cache capacity: `16` slots
- Catalog size: `128` items
- Episode limit: `1200` steps
- Request stream: sampled from item popularity (`self._popularity`)
- Reward:
  - Cache hit: `+item_cost`
  - Cache miss: `-item_cost`
  - Invalid/fallback manual behavior penalty: `-0.25` in specific full-cache fallback paths

## Data Source (Wikipedia vs Synthetic)

The environment can initialize item catalog/popularity from Wikimedia pageviews.

- Enabled by: `USE_WIKIPEDIA_DATA=true`
- Fetch endpoint pattern:
  - `https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/YYYY/MM/DD`
- It tries yesterday, then up to 7 days back.
- On success, `metadata.catalog_source` and `ui_summary` include `wikipedia:<date>`.
- On failure, it falls back to synthetic data (`catalog_source=synthetic`).

Field meaning in observation:
- `incoming_item_id`: index into internal catalog arrays.
- `incoming_popularity`: normalized probability from pageviews.
- `incoming_size`: deterministic synthetic size derived from title hash, then normalized.
- `incoming_cost`: rank-based synthetic cost (less popular ranks cost more), then normalized.

## API Endpoints

Main routes exposed by `server.app:app`:
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `GET /metadata`
- `GET /health`
- `GET /docs` (Swagger)
- `GET /redoc`
- `WS /ws`
- MCP endpoints (`/mcp`)

Note: `GET /` is not defined by default and returns `404`.

## Web UI (OpenEnv Playground)

The OpenEnv web playground is served at:
- `/web`

To enable it, set:
- `ENABLE_WEB_INTERFACE=true`

Run locally with env vars loaded:

```bash
set -a
source .env
set +a
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Open:
- `http://127.0.0.1:7860/web` (Playground)
- `http://127.0.0.1:7860/docs` (Swagger)

## Reset and Step Semantics

### Reset
`POST /reset` initializes a fresh episode and returns an initial observation.

Typical reset output characteristics:
- `step=0`
- `reward=0.0`
- empty cache (`cache_fill_ratio=0.0`, all `slot=... empty`)
- first incoming request already sampled

At reset, `ui_summary` may show `MISS` because `_last_hit` default is `False`; this is not a real miss event from a step yet.

### Step
`POST /step` applies one action and advances by one request.

Action model (`SmartCacheRlAction`):
- `evict_index: Optional[int]`
- `use_agent: bool`
- `agent_mode: Optional[str]` (`manual|lru|lfu|dqn`)

## How to Provide Eviction Action

### Manual mode
When cache is full and the incoming item is a miss, provide a slot index to evict:

```json
{
  "evict_index": 3,
  "use_agent": false
}
```

### Agent-driven mode
Let environment choose eviction:

```json
{
  "use_agent": true,
  "agent_mode": "dqn"
}
```

Also valid: `agent_mode: "lru"` or `"lfu"`.

Important behavior:
- If cache is not full, item is inserted directly (no eviction needed).
- If manual action omits `evict_index` while full, environment falls back to LRU and applies penalty.

## Observation Fields You Will Commonly Use

- `ui_summary`
- `ui_hit_rate`
- `ui_cache_snapshot`
- `ui_suggested_lru_slot`
- `ui_suggested_lfu_slot`
- `ui_suggested_dqn_slot`
- `ui_action_source`
- `ui_applied_evict_slot`
- `metadata.hit_rate`
- `metadata.catalog_source`

## Redis Integration (Optional)

Environment supports optional Redis writes for latest metrics.

- Enable: `USE_REDIS=true`
- Configure: `REDIS_URL=redis://...` or `rediss://...`

Status surfaces in observation:
- `ui_redis_status_line`
- `metadata.redis_status`
- `metadata.redis_latest`

If Redis fails, environment continues in memory mode.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

If needed:
```bash
pip install uvicorn numpy
```

## Running Locally

```bash
set -a
source .env
set +a
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

## Docker

Build:
```bash
docker build -t smart_cache_rl-env:latest -f Dockerfile .
```

Run:
```bash
docker run --rm -p 7860:7860 --env-file .env smart_cache_rl-env:latest
```

## Hugging Face Spaces / OpenEnv Push

From project root (with `openenv.yaml` present):

```bash
openenv push
```

Useful options:
- `--repo-id <namespace/repo>`
- `--private`
- `--base-image <image>`

## Project Structure

```
smart-cache-rl/
├── README.md
├── openenv.yaml
├── pyproject.toml
├── .env
├── agent.py
├── models.py
├── inference.py
├── client.py
└── server/
    ├── app.py
    └── smart_cache_rl_environment.py
```

## Quick Troubleshooting

- `/web` returns 404:
  - Ensure `ENABLE_WEB_INTERFACE=true` is exported before launching uvicorn.
- `/` returns 404:
  - Expected; use `/web` or `/docs`.
- `No module named uvicorn`:
  - Install in active venv: `pip install uvicorn`.
- Redis status shows `connection_failed_fallback_memory`:
  - Verify `REDIS_URL`, TLS scheme (`rediss://` if needed), and network accessibility.
