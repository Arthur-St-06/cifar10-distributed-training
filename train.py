import os
import torch
import torch.distributed as dist
from ultralytics import YOLO

def main():
    if torch.cuda.is_available():
        print("PyTorch is using a GPU")
        backend = "nccl"
        device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', 0))}")
    else:
        print("PyTorch is not using a GPU")
        backend = "gloo"
        device = torch.device("cpu")

    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(backend=backend)

    model = YOLO("yolov8n.yaml")
    model.to(device)

    results = model.train(
        data="config.yaml",
        epochs=500,
        device=device.index if device.type == "cuda" else "cpu",
        workers=4
    )

if __name__ == "__main__":
    main()
