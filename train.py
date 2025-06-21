import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleModel
from dataset import get_dataloader

def main():
    dist.init_process_group(backend="gloo")  # Use "nccl" if you have GPUs
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))  # fallback for torchrun
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    print(f"local rank: {local_rank}")

    model = SimpleModel().to(device)
    ddp_model = DDP(model, device_ids=None)

    dataloader = get_dataloader(batch_size=64, rank=rank, world_size=world_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=1e-3)

    print("starting training")

    for epoch in range(2):
        ddp_model.train()
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = ddp_model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            if batch % 100 == 0 and rank == 0:
                print(f"Epoch {epoch} Batch {batch} Loss {loss.item():.4f} Rank {rank}")

    print(f"1 finished training with local rank: {rank}")

    dist.destroy_process_group()

    print(f"2 finished training with local rank: {rank}")

if __name__ == "__main__":
    main()
