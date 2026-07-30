#!/usr/bin/env bash
# ───────────────────────────────────────────────
# find_gpu.sh — Find free GPUs on ECS servers
# ───────────────────────────────────────────────
# Usage:
#   ./code/fine-tuning/find_gpu.sh [options]
#
# Options:
#   --gpu-mem <GB>     Minimum GPU memory in GB (default: 24)
#   --min-gpus <N>     Minimum number of free GPUs (default: 1)
#   --jump <host>      Run via a jump host (default: uni-entry)
#   --update-config    Write top 3 servers to .ssh_config as uni-gpu1/2/3
#   --help, -h         Show this help
#
# Examples:
#   ./code/fine-tuning/find_gpu.sh
#   ./code/fine-tuning/find_gpu.sh --gpu-mem 48 --min-gpus 2
#   ./code/fine-tuning/find_gpu.sh --update-config
#
# What it does:
#   1. Checks known ECS GPU servers via SSH + nvidia-smi
#   2. Filters by GPU memory and free GPU count (util < 5%)
#   3. Sorts candidates (highest mem → lowest avg util → most free GPUs)
#   4. Prints results
#   5. With --update-config: writes the top 3 to .ssh_config
#      (replaceable via auto-generated markers)
#
# Dependencies:
#   - SSH access to uni-entry
#   - SSH keys loaded for the ECS servers
# ───────────────────────────────────────────────

set -euo pipefail

help() { sed -n '3,31p' "$0"; }
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then help; exit 0; fi

MIN_GPU_MEM=24
MIN_GPUS=1
JUMP_HOST="uni-entry"
UPDATE_CONFIG=false
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SSH_CONFIG="$SCRIPT_DIR/.ssh_config"

# Detect if we are running on the jump host (script was piped to /tmp/)
# and disable the jump to prevent infinite recursion.
if [[ "$SCRIPT_DIR" == "/tmp" ]] || [[ "$0" == /tmp/* ]]; then
  JUMP_HOST=""
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu-mem) MIN_GPU_MEM="$2"; shift 2 ;;
    --min-gpus) MIN_GPUS="$2"; shift 2 ;;
    --jump) JUMP_HOST="$2"; shift 2 ;;
    --update-config) UPDATE_CONFIG=true; shift ;;
    --help|-h) help; exit 0 ;;
    *)
      echo "Unknown option: $1"
      exit 1 ;;
  esac
done

SERVERS=(
  cuda-small0  cuda-small1
  cuda00 cuda01 cuda02 cuda03 cuda04 cuda05 cuda06 cuda07
  cuda08 cuda09 cuda10 cuda11 cuda12 cuda13 cuda14 cuda15
  cuda16 cuda17 cuda18 cuda19 cuda20 cuda21 cuda22 cuda23 cuda24
  gryphon red-tomatoes piccolo the-villa bordeaux
)

declare -A GPU_MEM
GPU_MEM[cuda-small0]=16   GPU_MEM[cuda-small1]=20
GPU_MEM[cuda00]=24        GPU_MEM[cuda01]=24
GPU_MEM[cuda02]=24        GPU_MEM[cuda03]=24
GPU_MEM[cuda04]=24        GPU_MEM[cuda05]=24
GPU_MEM[cuda06]=24        GPU_MEM[cuda07]=24
GPU_MEM[cuda08]=24        GPU_MEM[cuda09]=24
GPU_MEM[cuda10]=24        GPU_MEM[cuda11]=24
GPU_MEM[cuda12]=24        GPU_MEM[cuda13]=24
GPU_MEM[cuda14]=48        GPU_MEM[cuda15]=48
GPU_MEM[cuda16]=48        GPU_MEM[cuda17]=24
GPU_MEM[cuda18]=48        GPU_MEM[cuda19]=24
GPU_MEM[cuda20]=48        GPU_MEM[cuda21]=48
GPU_MEM[cuda22]=48        GPU_MEM[cuda23]=96
GPU_MEM[cuda24]=96
GPU_MEM[gryphon]=48       GPU_MEM[red-tomatoes]=48
GPU_MEM[piccolo]=24       GPU_MEM[the-villa]=48
GPU_MEM[bordeaux]=24

# If --jump is set, copy this script to the jump host and run it there
if [[ -n "$JUMP_HOST" ]]; then
  echo "Running on $JUMP_HOST..."
  REMOTE_ARGS=()
  [[ "$MIN_GPU_MEM" != 24 ]] && REMOTE_ARGS+=(--gpu-mem "$MIN_GPU_MEM")
  [[ "$MIN_GPUS" != 1 ]] && REMOTE_ARGS+=(--min-gpus "$MIN_GPUS")
  if $UPDATE_CONFIG; then
    # Run remotely but write to a temp file, then SCP it back locally
    ssh -F "$SCRIPT_DIR/.ssh_config" -o StrictHostKeyChecking=accept-new "$JUMP_HOST" \
      "cat > /tmp/find_gpu.sh && chmod +x /tmp/find_gpu.sh && CONFIG_OUT=/tmp/ssh_config_update /tmp/find_gpu.sh --update-config ${REMOTE_ARGS[*]}" \
      < "$0"
    scp -F "$SCRIPT_DIR/.ssh_config" -o StrictHostKeyChecking=accept-new "$JUMP_HOST":/tmp/ssh_config_update "$SSH_CONFIG"
    echo "Wrote config to $SSH_CONFIG"
  else
    ssh -F "$SCRIPT_DIR/.ssh_config" -o StrictHostKeyChecking=accept-new "$JUMP_HOST" \
      "cat > /tmp/find_gpu.sh && chmod +x /tmp/find_gpu.sh && /tmp/find_gpu.sh ${REMOTE_ARGS[*]}" \
      < "$0"
  fi
  exit $?
fi

echo "Scanning GPU servers (min ${MIN_GPU_MEM}GB GPU, min ${MIN_GPUS} free GPUs)..."
echo ""

unreachable=()
declare -a candidates

for server in "${SERVERS[@]}"; do
  if [[ "${GPU_MEM[$server]:-0}" -lt "$MIN_GPU_MEM" ]]; then
    continue
  fi

  if ! ssh -o ConnectTimeout=3 -o StrictHostKeyChecking=accept-new "$server" exit 2>/dev/null; then
    unreachable+=("$server")
    continue
  fi

  output=$(ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=accept-new "$server" \
    nvidia-smi --query-gpu=index,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || true)

  if [[ -z "$output" ]]; then
    continue
  fi

  free_count=0
  total_gpus=0
  total_util=0
  while IFS=',' read -r idx mem_mb util; do
    idx="$(echo "$idx" | xargs)"
    util="$(echo "$util" | xargs | sed 's/ %//')"
    total_gpus=$((total_gpus + 1))
    total_util=$((total_util + util))
    if [[ "$util" -lt 5 ]]; then
      free_count=$((free_count + 1))
    fi
  done <<< "$output"

  if [[ "$free_count" -ge "$MIN_GPUS" ]]; then
    avg_util=$((total_util / total_gpus))
    gpu_mem="${GPU_MEM[$server]}"
    candidates+=("$gpu_mem:$avg_util:$free_count:$server")
  fi
done

# Sort: highest GPU mem first, then lowest avg utilization, then most free GPUs
IFS=$'\n' sorted=($(sort -t: -k1 -rn -k2 -n -k3 -rn <<< "${candidates[*]}"))
unset IFS

# Display all candidates
for entry in "${sorted[@]}"; do
  IFS=':' read -r mem util free server <<< "$entry"
  echo "========================================"
  echo "  $server  (${mem}GB GPUs, avg util ${util}%)"
  echo "  Free GPUs: $free"
  echo ""
done

if [[ ${#sorted[@]} -eq 0 ]]; then
  echo "No servers found with ${MIN_GPUS}+ free GPU(s) of ${MIN_GPU_MEM}GB or more."
  if [[ ${#unreachable[@]} -gt 0 ]]; then
    echo ""
    echo "Could not reach ${#unreachable[@]} server(s):"
    for s in "${unreachable[@]}"; do
      echo "  - $s"
    done
  fi
  exit 1
fi

if [[ ${#unreachable[@]} -gt 0 ]]; then
  echo "(Could not reach ${#unreachable[@]} server(s): ${unreachable[*]})"
fi

# --update-config: write top 3 to .ssh_config
if $UPDATE_CONFIG; then
  CONFIG_OUT="${CONFIG_OUT:-$SSH_CONFIG}"
  count=0
  BLOCK=""
  for entry in "${sorted[@]}"; do
    IFS=':' read -r mem util free server <<< "$entry"
    count=$((count + 1))
    BLOCK+="Host uni-gpu${count}"$'\n'
    BLOCK+="    HostName $server.ecs.vuw.ac.nz"$'\n'
    BLOCK+="    User thompsjame1"$'\n'
    BLOCK+="    ProxyJump uni-entry"$'\n'
    BLOCK+="    IdentityFile ~/.ssh/id_rsa"$'\n'
    BLOCK+="    ForwardAgent yes"$'\n'
    BLOCK+=$'\n'
    if [[ "$count" -ge 3 ]]; then
      break
    fi
  done

  if [[ -f "$CONFIG_OUT" ]] && grep -q "^# BEGIN AUTO-GENERATED uni-gpu" "$CONFIG_OUT"; then
    # Replace content between markers
    awk -v block="$BLOCK" '
      /^# BEGIN AUTO-GENERATED uni-gpu/ { print; printing = 1; next }
      /^# END AUTO-GENERATED uni-gpu/   { print block; printing = 0; next }
      !printing                           { print }
    ' "$CONFIG_OUT" > "${CONFIG_OUT}.tmp" && mv "${CONFIG_OUT}.tmp" "$CONFIG_OUT"
  else
    # Append new block to existing file or create new
    {
      [[ -f "$CONFIG_OUT" ]] && cat "$CONFIG_OUT"
      echo "# BEGIN AUTO-GENERATED uni-gpu"
      echo -n "$BLOCK"
      echo "# END AUTO-GENERATED uni-gpu"
    } > "${CONFIG_OUT}.tmp" && mv "${CONFIG_OUT}.tmp" "$CONFIG_OUT"
  fi
  echo "Wrote ${count} host(s) to $CONFIG_OUT"
fi