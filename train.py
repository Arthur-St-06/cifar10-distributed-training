import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleModel
from dataset import get_dataloader
import os

def main():
    print("---------------1")

    # Map OpenMPI → PyTorch var names
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"]        = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"]  = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"]  = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", os.environ.get("HOSTNAME", "localhost"))
    os.environ.setdefault("MASTER_PORT", "12345")

    print("---------------2")

    dist.init_process_group(backend="gloo")  # For CPU; use "nccl" if you switch to GPU

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Safe fallback

    print("---------------3")

    device = torch.device("cpu")  # Explicitly use CPU for your test

    print(f"[Rank {rank}] Initialized process group on {device}")

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
        # make 100 % sure it reaches the logs
        traceback.print_exc()
        sys.stderr.flush(); sys.stdout.flush()
        raise
