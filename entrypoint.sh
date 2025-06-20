#!/bin/bash
NODE_RANK=$(echo "$HOSTNAME" | grep -oE '[0-9]+$')

echo "Starting training with node_rank=$NODE_RANK"

torchrun \
  --nproc-per-node=1 \
  --nnodes=2 \
  --node_rank="$NODE_RANK" \
  --rdzv_backend=c10d \
  --rdzv_endpoint=yolo-ddp-0.yolo-ddp.default.svc.cluster.local:29500 \
  train.py
