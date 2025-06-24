import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleModel
from dataloader import get_dataloader
import contextlib

import os
import yaml
import time

def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"]        = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"]  = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"]  = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
    else:
        print("RANK not found, assuming standalone debug mode.")
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    os.environ.setdefault("MASTER_PORT", str(config["ddp"]["port"]))

    dist.init_process_group(backend=config["ddp"]["backend"])

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(config["training"]["device"])

    print(f"[Rank {rank}] Initialized process group on {device}")

    model = SimpleModel().to(device)
    ddp_model = DDP(model)

    dataloader = get_dataloader(batch_size=config["training"]["batch_size"], rank=rank, world_size=world_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(ddp_model.parameters(), lr=config["training"]["lr"])

    if config["wandb"]["use"] and rank == 0:
        import wandb
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="mnist", config={"lr": config["training"]["lr"]})

    accum_steps = config["training"]["accumulation_steps"]
    global_step = 0

    # Start timing
    if rank == 0:
        start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        ddp_model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            # ---------------------------------------------------------------
            # 1. choose whether to sync this mini-batch
            sync_context = (
                contextlib.nullcontext()
                if (batch_idx + 1) % accum_steps == 0
                else ddp_model.no_sync()    # skip gradient all-reduce this step
            )
            with sync_context:
                # -----------------------------------------------------------
                # 2. forward / backward – scale loss so total gradient stays the same
                output = ddp_model(x)
                loss   = loss_fn(output, y) / accum_steps
                loss.backward()
            # ---------------------------------------------------------------
            # 3. perform the real optimiser step every `accum_steps`
            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if rank == 0 and global_step % config["wandb"]["log_interval"] == 0:
                    wandb.log({"loss": loss.item() * accum_steps,  # undo scaling
                            "step": global_step,
                            "epoch": epoch})
                    print(f"[Rank {rank}] Epoch {epoch} "
                        f"Step {global_step} Loss {loss.item()*accum_steps:.4f}")

    # Wait for all workers to finish before measuring time
    dist.barrier()

    if rank == 0:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"[Rank 0] Total training time: {total_time:.2f} seconds")

    if config["wandb"]["use"] and rank == 0:
        wandb.finish()

    print(f"[Rank {rank}] Training complete.")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
