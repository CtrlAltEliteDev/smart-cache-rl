# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Smart Cache RL environment."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.responses import HTMLResponse

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

# Force-disable Gradio/OpenEnv web wrapper; we serve our own UI at `/`.
os.environ["ENABLE_WEB_INTERFACE"] = "false"

app = create_app(
    SmartCacheRlEnvironment,
    SmartCacheRlAction,
    SmartCacheRlObservation,
    env_name="smart_cache_rl",
    max_concurrent_envs=1,
)

_UI_HTML_PATH = Path(__file__).parent / "static" / "index.html"


def _load_ui_html() -> str:
    if not _UI_HTML_PATH.exists():
        return "<h1>UI file missing</h1><p>Expected server/static/index.html</p>"
    return _UI_HTML_PATH.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def custom_ui_root() -> HTMLResponse:
    return HTMLResponse(_load_ui_html())


@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def custom_ui_alias() -> HTMLResponse:
    return HTMLResponse(_load_ui_html())


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for direct execution via uv run or python -m."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
