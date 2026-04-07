---
title: Smart Cache Rl Environment Server
emoji: 🎧
colorFrom: pink
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Smart Cache Rl Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Smart Cache Rl environment is through the `SmartCacheRlEnv` class:

```python
from smart_cache_rl import SmartCacheRlAction, SmartCacheRlEnv

try:
    # Create environment from Docker image
    smart_cache_rlenv = SmartCacheRlEnv.from_docker_image("smart_cache_rl-env:latest")

    # Reset
    result = smart_cache_rlenv.reset()
    print(f"Reset: {result.observation.echoed_message}")

    # Send multiple messages
    messages = ["Hello, World!", "Testing echo", "Final message"]

    for msg in messages:
        result = smart_cache_rlenv.step(SmartCacheRlAction(message=msg))
        print(f"Sent: '{msg}'")
        print(f"  → Echoed: '{result.observation.echoed_message}'")
        print(f"  → Length: {result.observation.message_length}")
        print(f"  → Reward: {result.reward}")

finally:
    # Always clean up
    smart_cache_rlenv.close()
```

That's it! The `SmartCacheRlEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t smart_cache_rl-env:latest -f Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

### Optional Redis Integration (HF Space compatible)

Redis is optional and defaults to in-memory mode.

- `USE_REDIS=true` enables Redis writes
- `REDIS_URL=<your_redis_url>` points to an external Redis instance
- Free cloud option: Upstash Redis (`rediss://default:<password>@<host>:6379`)

If Redis is unavailable, the environment automatically falls back to in-memory mode and still serves `/web` and `/health`.
When Redis is enabled, UI observation includes `ui_redis_status_line` and `metadata.redis_latest` so you can verify cloud Redis is being read/written.

## Environment Details

## Agent, LRU, and LFU

### How the agent is connected

- The RL agent is implemented in [agent.py](/home/riya/smart-cache-rl/smart_cache_rl/agent.py) as `DQNCacheAgent`.
- It is connected in two execution paths:
  - **Environment/UI path**: [server/smart_cache_rl_environment.py](/home/riya/smart-cache-rl/smart_cache_rl/server/smart_cache_rl_environment.py) supports `use_agent=true` with `agent_mode=dqn|lru|lfu|manual`.
  - **Inference path**: [inference.py](/home/riya/smart-cache-rl/smart_cache_rl/inference.py) supports `POLICY_MODE=dqn|lru|lfu|llm`.
- In UI, agent signals are shown in observation fields like:
  - `ui_agent_mode`
  - `ui_suggested_dqn_slot`
  - `ui_action_source`
  - `ui_applied_evict_slot`

### Why RL can be better than LRU/LFU

- **LRU** only uses recency and ignores item cost/size/popularity shifts.
- **LFU** only uses frequency and adapts slowly when traffic distribution changes.
- **RL (DQN)** uses a richer state: recency + frequency + size + cost + incoming request features.
- Because reward is tied to miss/hit cost, RL learns to keep items that are expensive to miss, not just recent/frequent.
- This generally improves weighted hit quality under skewed (Zipf-like) workloads compared with fixed heuristics.

### Practical expectation

- On simple stable workloads, LRU/LFU may be competitive.
- On mixed-cost, non-uniform request streams, RL is expected to outperform by learning tradeoffs heuristics cannot represent directly.

### Action
**SmartCacheRlAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**SmartCacheRlObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Smart Cache Rl environment server running, you can connect directly:

```python
from smart_cache_rl import SmartCacheRlEnv

# Connect to existing server
smart_cache_rlenv = SmartCacheRlEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = smart_cache_rlenv.reset()
result = smart_cache_rlenv.step(SmartCacheRlAction(message="Hello!"))
```

Note: When connecting to an existing server, `smart_cache_rlenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from smart_cache_rl import SmartCacheRlAction, SmartCacheRlEnv

# Connect with context manager (auto-connects and closes)
with SmartCacheRlEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(SmartCacheRlAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    SmartCacheRlEnvironment,  # Pass class, not instance
    SmartCacheRlAction,
    SmartCacheRlObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from smart_cache_rl import SmartCacheRlAction, SmartCacheRlEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with SmartCacheRlEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(SmartCacheRlAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/smart_cache_rl_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
smart_cache_rl/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # SmartCacheRlEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── smart_cache_rl_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
