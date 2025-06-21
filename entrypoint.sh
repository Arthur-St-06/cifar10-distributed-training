#!/bin/bash
set -e

echo "HOSTNAME is: $HOSTNAME"

# Extract numeric node rank from pod name
NODE_RANK=$(echo "$HOSTNAME" | grep -o '[0-9]*$')

# Validate
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
  echo "Failed to extract a valid node rank from HOSTNAME: $HOSTNAME"
  exit 1
fi

echo "Starting training with node_rank=$NODE_RANK"

# MPIJob will use service DNS like yolo-ddp-mpi-launcher.default.svc
RDZV_ENDPOINT="yolo-ddp-mpi-launcher.default.svc:29500"

torchrun \
  --nproc-per-node=1 \
  --nnodes=2 \
  --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_ENDPOINT" \
  train.py
