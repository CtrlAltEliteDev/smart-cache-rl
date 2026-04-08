# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Smart Cache RL environment."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import ValidationError

try:
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.serialization import deserialize_action, serialize_observation
    from openenv.core.env_server.types import ResetRequest, ResetResponse, StepRequest, StepResponse, State
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

# OpenEnv's default HTTP /reset, /step, and /state handlers create a *new* environment on every
# request and close it afterward, so the cache never accumulates across steps (only slot 0 fills).
# Replace them with a single long-lived instance for REST clients and the static dashboard.
_http_env_lock = threading.Lock()
_http_env: SmartCacheRlEnvironment | None = None


def _http_env_instance() -> SmartCacheRlEnvironment:
    global _http_env
    if _http_env is None:
        _http_env = SmartCacheRlEnvironment()
    return _http_env


def _serialize_step_response(observation: SmartCacheRlObservation) -> dict:
    """OpenEnv's serialize_observation drops `metadata`; the dashboard expects it under observation."""
    payload = serialize_observation(observation)
    meta = getattr(observation, "metadata", None) or {}
    if meta:
        payload["observation"] = {**payload["observation"], "metadata": meta}
    return payload


def _remove_stateless_simulation_routes(application: FastAPI) -> None:
    kept: list = []
    for route in list(application.router.routes):
        path = getattr(route, "path", "")
        methods = getattr(route, "methods", None) or set()
        if path == "/reset" and "POST" in methods:
            continue
        if path == "/step" and "POST" in methods:
            continue
        if path == "/state" and "GET" in methods:
            continue
        kept.append(route)
    application.router.routes = kept


_remove_stateless_simulation_routes(app)


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
def persistent_reset(_request: ResetRequest = Body(default_factory=ResetRequest)) -> ResetResponse:
    # SmartCacheRlEnvironment.reset() ignores seed for now; _request kept for API compatibility.
    with _http_env_lock:
        observation = _http_env_instance().reset()
    return ResetResponse(**_serialize_step_response(observation))


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
def persistent_step(request: StepRequest) -> StepResponse:
    try:
        action = deserialize_action(request.action, SmartCacheRlAction)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors()) from e
    with _http_env_lock:
        observation = _http_env_instance().step(action)
    return StepResponse(**_serialize_step_response(observation))


@app.get("/state", response_model=State, tags=["State Management"])
def persistent_state() -> State:
    with _http_env_lock:
        return _http_env_instance().state

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


@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def hf_web_alias() -> HTMLResponse:
    # Hugging Face Space probes `/web` for compatibility with OpenEnv-style UIs.
    # Serve the same dashboard HTML as `/` and `/ui`.
    return HTMLResponse(_load_ui_html())


@app.get("/web/", response_class=HTMLResponse, include_in_schema=False)
async def hf_web_alias_slash() -> HTMLResponse:
    return HTMLResponse(_load_ui_html())


def _run_server(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    """Validator-friendly callable entrypoint for console scripts."""
    _run_server(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _run_server(host="0.0.0.0", port=args.port)
