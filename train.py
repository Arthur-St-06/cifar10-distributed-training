import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleModel
from dataloader import get_dataloader
import os

def main():
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"]        = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"]  = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"]  = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    else:
        print("RANK not found, assuming standalone debug mode.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    os.environ.setdefault("MASTER_PORT", "29500")

    dist.init_process_group(backend="gloo")  # For CPU; use "nccl" if you switch to GPU

    rank = dist.get_rank()
    
    world_size = dist.get_world_size()

    device = torch.device("cpu")

    print(f"[Rank {rank}] Initialized process group n {device}")

    model = SimpleModel().to(device)
    ddp_model = DDP(model)

    dataloader = get_dataloader(batch_size=64, rank=rank, world_size=world_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    for epoch in range(2):
        ddp_model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                print(f"[Rank {rank}] Epoch {epoch} Batch {batch} Loss: {loss.item():.4f}")

    print(f"[Rank {rank}] Training complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    import traceback, sys
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        sys.stderr.flush(); sys.stdout.flush()
        raise
