# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.10-slim

WORKDIR /app/env

COPY server/requirements.txt /app/env/server/requirements.txt
RUN pip install --no-cache-dir -r /app/env/server/requirements.txt

COPY . /app/env

# Set PYTHONPATH so imports work correctly
ENV PYTHONPATH="/app/env"
# Enable OpenEnv Gradio web UI at /web
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=2)" || exit 1

# Run the FastAPI server
# The module path is constructed to work with the /app/env structure
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
