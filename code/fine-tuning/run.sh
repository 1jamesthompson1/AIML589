#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────
# run.sh — Run fine-tuning or serving on a remote GPU host
# ───────────────────────────────────────────────
# Usage:
#   ./code/fine-tuning/run.sh serve    <ssh-host> [--port PORT] [-- serve.py args...]
#   ./code/fine-tuning/run.sh finetune <ssh-host> [-- finetune.py args...]
#
# Examples:
#   ./code/fine-tuning/run.sh finetune uni-gpu1 \
#       -- --dataset modal_response --subpopulation cluster_0 --upload-to-hf
#
#   ./code/fine-tuning/run.sh serve uni-gpu1 --port 8087 \
#       -- --model Qwen/Qwen3.6-27B --dataset modal_response --subpopulation cluster_0
#
# Environment:
#   REMOTE_DIR   Remote working directory (default: llm-ft)
# ───────────────────────────────────────────────

help() { sed -n '7,16p' "$0"; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then help; exit 0; fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_CONFIG="$SCRIPT_DIR/.ssh_config"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 {serve|finetune} <ssh-host> [options] [-- args...]"
  echo "  $0 --help for details"
  exit 1
fi

MODE="$1"; shift
HOST="$1"; shift

case "$MODE" in
  serve|finetune) ;;
  *) echo "Unknown mode: $MODE (use 'serve' or 'finetune')"; exit 1 ;;
esac

# Defaults
PORT="${PORT:-8080}"
REMOTE_ARGS=()

# Parse options before --
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --) shift; REMOTE_ARGS+=("$@"); break ;;
    *) REMOTE_ARGS+=("$1"); shift ;;
  esac
done

# Pick script based on mode
case "$MODE" in
  serve)   SCRIPT="serve.py" ;;
  finetune) SCRIPT="finetune.py" ;;
  *) echo "Unknown mode: $MODE (use 'serve' or 'finetune')"; exit 1 ;;
esac

REMOTE_DIR="${REMOTE_DIR:-llm-ft}"
ENV_FILE="finetune.env"
REMOTE_FULL="\$HOME/$REMOTE_DIR"
REMOTE_SCP="~/$REMOTE_DIR"

echo "==> Setting up remote directory on $HOST..."
ssh -F "$SSH_CONFIG" "$HOST" "mkdir -p \"$REMOTE_FULL\""

# Interrupted runs orphan uv ephemeral envs in /tmp (each several GB of
# torch/vllm), eventually filling the disk. Clear any not referenced by a
# live process before starting.
ssh -F "$SSH_CONFIG" "$HOST" 'for d in /tmp/.tmp*/environments-v2/serve-*; do
  [ -e "$d" ] || continue
  if ! grep -l "$d" /proc/[0-9]*/environ 2>/dev/null | grep -q .; then
    rm -rf "$(dirname "$(dirname "$d")")"
  fi
done'

echo "==> Copying files to $HOST..."
scp -F "$SSH_CONFIG" "$SCRIPT_DIR/$SCRIPT" "$HOST":"$REMOTE_SCP"/"$SCRIPT"
# Copy template for model card generation
if [ "$MODE" = "finetune" ]; then
  scp -F "$SSH_CONFIG" "$SCRIPT_DIR/MODEL_DATACARD_TEMPLATE.md" "$HOST":"$REMOTE_SCP"/MODEL_DATACARD_TEMPLATE.md
fi

# Sync env vars to remote
ENV_VARS=("HF_TOKEN" "HF_ORG" "HF_COLLECTION")
TMPENV=$(mktemp)
for key in "${ENV_VARS[@]}"; do
  val=$(grep -s "^${key}=" "$SCRIPT_DIR/../../.env" | head -1 | cut -d= -f2- | tr -d "\"'")
  if [ -n "$val" ]; then
    echo "${key}=${val}" >> "$TMPENV"
  fi
done
scp -F "$SSH_CONFIG" "$TMPENV" "$HOST":"$REMOTE_SCP"/"$ENV_FILE"
rm -f "$TMPENV"

# Build remote command prefix
REMOTE_PREFIX="export PATH=\"\$HOME/.local/bin:\$PATH\";"
REMOTE_PREFIX+=" for d in /usr/local/cuda /opt/cuda /usr/lib/cuda; do"
REMOTE_PREFIX+="   [ -x \"\$d/bin/nvcc\" ] && export CUDA_HOME=\"\$d\" && export PATH=\"\$d/bin:\$PATH\" && break;"
REMOTE_PREFIX+=" done;"
REMOTE_PREFIX+=" command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh;"
REMOTE_PREFIX+=" cd \"$REMOTE_FULL\" && set -a && . ./$ENV_FILE && set +a"

if [ "$MODE" = "serve" ]; then
  # ── Serve mode: find free port, start vLLM, set up tunnel ──
  echo "==> Finding free port on $HOST..."
  INT_PORT=$(ssh -F "$SSH_CONFIG" "$HOST" "python3 -c 'import socket; s=socket.socket(); s.bind((\"\",0)); print(s.getsockname()[1]); s.close()'")
  echo "     internal port: $INT_PORT"

  echo "==> Running serve.py on $HOST (internal :$INT_PORT, external :$PORT)..."
  REMOTE_CMD="$REMOTE_PREFIX && uv run $SCRIPT --port $INT_PORT ${REMOTE_ARGS[*]}"

  ssh -F "$SSH_CONFIG" -t "$HOST" "$REMOTE_CMD" &
  REMOTE_PID=$!
  sleep 3

  echo "==> Setting up SSH tunnel (localhost:$PORT -> $HOST:$INT_PORT)..."
  ssh -F "$SSH_CONFIG" -N -L "$PORT:localhost:$INT_PORT" "$HOST" &
  TUNNEL_PID=$!

  echo ""
  echo "============================================"
  echo "  Server running!"
  echo "  Access via: http://localhost:$PORT/v1/models"
  echo "  Tunnel: localhost:$PORT -> $HOST:$INT_PORT"
  echo "  Run eval: uv run evaluate.py ... --port $PORT"
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
else
  # ── Finetune mode: run in foreground ──
  echo "==> Running $(basename "$SCRIPT") on $HOST..."
  REMOTE_CMD="$REMOTE_PREFIX && uv run $SCRIPT ${REMOTE_ARGS[*]}"

  ssh -F "$SSH_CONFIG" -t "$HOST" "$REMOTE_CMD"

  echo ""
  echo "============================================"
  echo "  Fine-tuning complete!"
  echo "============================================"
fi
