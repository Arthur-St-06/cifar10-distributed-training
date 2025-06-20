#!/bin/bash
set -e

echo "HOSTNAME is: $HOSTNAME"

# Extract numeric node rank
NODE_RANK=$(echo "$HOSTNAME" | grep -o '[0-9]*$')

# Validate
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
  echo "Failed to extract a valid node rank from HOSTNAME: $HOSTNAME"
  exit 1
fi

echo "Starting training with node_rank=$NODE_RANK"

torchrun \
  --nproc-per-node=1 \
  --nnodes=2 \
  --node_rank="0" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=yolo-ddp-0.yolo-ddp.default.svc.cluster.local:29500 \
  train.py
