#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────
# serve.sh — Deploy and run serve_model.py on a remote GPU host
# ───────────────────────────────────────────────
# Usage:
#   ./code/fine-tuning/serve.sh <ssh-host> [options] [serve_model.py args...]
#
# Arguments:
#   <ssh-host>       SSH hostname or alias (from .ssh_config)
#
# Options:
#   -P, --external-port PORT   Port on your laptop (default: 8080)
#                              An SSH tunnel forwards this to the remote.
#
# All other args are passed through to serve_model.py.
# The --port flag is injected automatically (internal random port).
#
# Examples:
#   ./code/fine-tuning/serve.sh uni-gpu1 -P 8087 --model Qwen/Qwen3.6-27B-FP8
#   ./code/fine-tuning/serve.sh vast-gpu -P 8080 --model Qwen/Qwen2.5-7B
#
# Environment:
#   REMOTE_DIR   Remote working directory (default: llm-ft)
#
# What it does:
#   1. Creates remote working directory
#   2. Copies serve_model.py to the remote host
#   3. Syncs env vars (HF_TOKEN from ../../.env) to finetune.env
#   4. Finds a free port on the remote
#   5. Installs uv if missing, runs serve_model.py on the remote (background)
#   6. Sets up SSH tunnel from localhost:<external-port> to remote:<internal-port>
#   7. Cleans up tunnel and remote process on exit
# ───────────────────────────────────────────────

help() { sed -n '3,30p' "$0"; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then help; exit 0; fi

SSH_CONFIG=".ssh_config"

HOST="${1:?Usage: $0 <ssh-host> [options] [serve_model.py args...]}"
shift

# Defaults
EXT_PORT=8080
MODEL_ARGS=()

# Parse serve.sh options vs serve_model.py args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -P|--external-port) EXT_PORT="$2"; shift 2 ;;
    -P=*|--external-port=*) EXT_PORT="${1#*=}"; shift ;;
    *) MODEL_ARGS+=("$1"); shift ;;
  esac
done

SCRIPT="serve_model.py"
REMOTE_DIR="${REMOTE_DIR:-llm-ft}"
ENV_FILE="finetune.env"
REMOTE_FULL="\$HOME/$REMOTE_DIR"
REMOTE_SCP="~/$REMOTE_DIR"

echo "==> Setting up remote directory on $HOST..."
ssh -F "$SSH_CONFIG" "$HOST" "mkdir -p \"$REMOTE_FULL\""

echo "==> Copying serve script to $HOST..."
scp "$SCRIPT" "$HOST":"$REMOTE_SCP"/"$SCRIPT"

# Sync only specified env vars to remote
ENV_VARS=("HF_TOKEN")
TMPENV=$(mktemp)
for key in "${ENV_VARS[@]}"; do
  val=$(grep -s "^${key}=" ../../.env | head -1 | cut -d= -f2- | tr -d "\"'")
  if [ -n "$val" ]; then
    echo "${key}=${val}" >> "$TMPENV"
  fi
done
scp -F "$SSH_CONFIG" "$TMPENV" "$HOST":"$REMOTE_SCP"/"$ENV_FILE"
rm -f "$TMPENV"

# Find a free port on the remote for vLLM to bind to
echo "==> Finding free port on $HOST..."
INT_PORT=$(ssh -F "$SSH_CONFIG" "$HOST" \
  "python3 -c \"import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()\"")
echo "     internal port: $INT_PORT"

# Strip any --port from MODEL_ARGS and inject our internal port
CLEAN_ARGS=()
skip_next=false
for arg in "${MODEL_ARGS[@]}"; do
  if $skip_next; then skip_next=false; continue; fi
  if [[ $arg == --port=* ]]; then continue; fi
  if [[ $arg == --port ]]; then skip_next=true; continue; fi
  CLEAN_ARGS+=("$arg")
done
set -- "${CLEAN_ARGS[@]}"

echo "==> Running serve_model.py on $HOST (internal :$INT_PORT, external :$EXT_PORT)..."
REMOTE_CMD="export PATH=\"\$HOME/.local/bin:\$PATH\"; for d in /usr/local/cuda /opt/cuda /usr/lib/cuda; do [ -x \"\$d/bin/nvcc\" ] && export CUDA_HOME=\"\$d\" && export PATH=\"\$d/bin:\$PATH\" && break; done; command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh; cd \"$REMOTE_FULL\" && set -a && . ./$ENV_FILE && set +a && uv run $SCRIPT --port $INT_PORT $*"

ssh -F "$SSH_CONFIG" -t "$HOST" "$REMOTE_CMD" &
REMOTE_PID=$!
sleep 3

echo "==> Setting up SSH tunnel (localhost:$EXT_PORT -> $HOST:$INT_PORT)..."
ssh -F "$SSH_CONFIG" -N -L "$EXT_PORT:localhost:$INT_PORT" "$HOST" &
TUNNEL_PID=$!

echo ""
echo "============================================"
echo "  Server running!"
echo "  Access via: http://localhost:$EXT_PORT/v1/models"
echo "  Tunnel: localhost:$EXT_PORT -> $HOST:$INT_PORT"
echo "  Run eval: uv run evaluate_model.py ... --port $EXT_PORT"
echo "  Press Ctrl+C to stop"
echo "============================================"
echo ""

cleanup() {
  echo ""; echo "==> Shutting down..."
  kill $TUNNEL_PID 2>/dev/null
  kill $REMOTE_PID 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM

wait $REMOTE_PID 2>/dev/null
cleanup