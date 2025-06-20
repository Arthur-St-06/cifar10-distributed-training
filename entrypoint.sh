#!/bin/bash
set -e

# Extract the numeric suffix from the hostname (e.g., yolo-ddp-0 → 0)
NODE_RANK=$(echo "$HOSTNAME" | grep -o '[0-9]*$')

# Validate it is a number
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
  echo "Failed to extract a valid node rank from HOSTNAME: $HOSTNAME"
  exit 1
fi

echo "Starting training with node_rank=$NODE_RANK"

torchrun \
  --nproc-per-node=1 \
  --nnodes=2 \
  --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=yolo-ddp-0.yolo-ddp.default.svc.cluster.local:29500 \
  train.py
