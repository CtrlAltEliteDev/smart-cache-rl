# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Smart Cache Rl Environment.

This module creates an HTTP server that exposes the SmartCacheRlEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SmartCacheRlAction, SmartCacheRlObservation
    from .smart_cache_rl_environment import SmartCacheRlEnvironment
except (ModuleNotFoundError, ImportError):
    from models import SmartCacheRlAction, SmartCacheRlObservation
    from server.smart_cache_rl_environment import SmartCacheRlEnvironment

import gradio as gr


def build_custom_ui(
    web_manager, action_fields, metadata, is_chat_env, title, quick_start_md
):
    """Custom UI tab: project-focused, dark-themed guide for this environment."""
    custom_css = """
    :root, .gradio-container {
      --body-background-fill: #0a0f1f !important;
      --body-background-fill-dark: #0a0f1f !important;
      --background-fill-primary: #0f1730 !important;
      --background-fill-secondary: #121c3b !important;
      --block-background-fill: #121c3b !important;
      --block-border-color: #2e3f6f !important;
      --color-accent: #6ea8ff !important;
      --color-accent-soft: #1f2f59 !important;
      --body-text-color: #e6ecff !important;
      --input-background-fill: #101a35 !important;
    }
    html, body {
      background: #0a0f1f !important;
      color: #e6ecff !important;
    }
    body, .gradio-container, .app {
      background: #0a0f1f !important;
      color: #e6ecff !important;
    }
    .gradio-container .tabs, .gradio-container .tabitem {
      background: #0f1730 !important;
      color: #e6ecff !important;
    }
    .gradio-container button, .gradio-container input, .gradio-container textarea {
      background: #121c3b !important;
      color: #e6ecff !important;
      border-color: #2e3f6f !important;
    }
    .gradio-container .prose, .gradio-container .markdown {
      color: #dbe5ff !important;
    }
    .sc-shell {
      background: linear-gradient(160deg, #0a0f1f 0%, #131a2f 100%);
      border: 1px solid #2a3558;
      border-radius: 14px;
      padding: 18px;
      color: #e6ecff;
      margin-bottom: 14px;
    }
    .sc-shell h2, .sc-shell h3 { color: #9ec5ff; margin: 0 0 10px 0; }
    .sc-muted { color: #a8b3d1; }
    .sc-chip {
      display: inline-block;
      margin: 4px 8px 4px 0;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #3a4a7a;
      color: #dbe7ff;
      background: #1b2747;
      font-size: 12px;
    }
    .sc-code {
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      background: #0d1530;
      border: 1px solid #2f3d69;
      border-radius: 10px;
      padding: 10px;
      color: #d8e3ff;
      white-space: pre-wrap;
    }
    """
    with gr.Blocks(css=custom_css) as demo:
        gr.HTML(
            """
<script>
(() => {
  const applyDark = () => {
    document.documentElement.classList.add("dark");
    document.body.classList.add("dark");
    document.documentElement.style.background = "#0a0f1f";
    document.body.style.background = "#0a0f1f";
    document.documentElement.style.color = "#e6ecff";
    document.body.style.color = "#e6ecff";
  };
  applyDark();
  setTimeout(applyDark, 50);
  setTimeout(applyDark, 400);
})();
</script>
"""
        )
        gr.Markdown(
            """
<div class="sc-shell">
  <h2>Smart Cache RL</h2>
  <p class="sc-muted">
    This project trains/evaluates cache eviction decisions so hit-rate beats classic LRU/LFU.
    Incoming requests are sampled from a Zipf-like distribution, and each item has size, cost, recency, and frequency signals.
  </p>
  <span class="sc-chip">Goal: maximize hit-rate</span>
  <span class="sc-chip">Action: choose evict slot</span>
  <span class="sc-chip">Reward: +cost on hit / -cost on miss</span>
</div>
"""
        )
        gr.Markdown(
            """
<div class="sc-shell">
  <h3>How It Works</h3>
  <p class="sc-muted">1) Click <b>Reset</b> in Playground to start a fresh episode.</p>
  <p class="sc-muted">2) Click <b>Step</b> repeatedly with manual action JSON or auto-agent mode.</p>
  <p class="sc-muted">3) Read <b>ui_summary</b>, <b>ui_hit_rate</b>, and <b>ui_cache_snapshot</b> from observation.</p>
</div>
"""
        )
        gr.Markdown(
            """
<div class="sc-shell">
  <h3>Action JSON Examples (copy into Playground Step action)</h3>
  <div class="sc-code">{"use_agent": true, "agent_mode": "dqn"}</div>
  <br/>
  <div class="sc-code">{"use_agent": true, "agent_mode": "lru"}</div>
  <br/>
  <div class="sc-code">{"evict_index": 3, "use_agent": false}</div>
</div>
"""
        )
        gr.Markdown(
            """
<div class="sc-shell">
  <h3>Agent Relationship</h3>
  <p class="sc-muted">
    The environment is the simulator. The agent chooses actions.
    In UI, <b>use_agent=true</b> lets environment auto-pick actions using <b>agent_mode</b> (dqn/lru/lfu).
    In scripts, <code>inference.py</code> can run DQN or LLM-driven policies.
  </p>
</div>
"""
        )
    return demo


# Create the app with web interface and README integration
app = create_app(
    SmartCacheRlEnvironment,
    SmartCacheRlAction,
    SmartCacheRlObservation,
    env_name="smart_cache_rl",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
    gradio_builder=build_custom_ui,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m smart_cache_rl.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn smart_cache_rl.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
