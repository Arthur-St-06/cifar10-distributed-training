import os
import torch
import torch.distributed as dist
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        print("PyTorch is using a GPU")
    else:
        print("PyTorch is not using a GPU")

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend="nccl")

    model = YOLO("yolov8n.yaml")

    results = model.train(
        data="config.yaml",
        epochs=500,
        device=local_rank,
        workers=4
    )

if __name__ == "__main__":
    main()
