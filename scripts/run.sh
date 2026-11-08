#!/usr/bin/env bash
# Start Layer Lab on http://$HOST:$PORT (default 127.0.0.1:7890). Behind Tailscale, for example:
#   tailscale serve --bg --set-path /layerlab http://127.0.0.1:7890
set -euo pipefail
cd "$(dirname "$0")/.."; . scripts/env.sh
cd app && exec "$LAYERLAB_VENV/bin/python" -m uvicorn server:app --host "${HOST:-127.0.0.1}" --port "${PORT:-7890}"
