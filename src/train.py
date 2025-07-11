import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model import SimpleModel
from download_cifar10_dataset import download_data
from dataloader import get_dataloader
import contextlib
import yaml
import time

from prometheus_client import start_http_server, Gauge

import boto3
import datetime
import tempfile

gpu_mem_usage = Gauge("gpu_memory_usage_mb", "GPU memory allocated (MB)")
loss_gauge = Gauge("training_loss", "Training loss")

def save_ckpt(state, ckpt_path, ckpt_cfg):
    torch.save(state, ckpt_path)
    print(f"Checkpoint saved locally at {ckpt_path}")

    s3 = boto3.client("s3")
    key = ckpt_cfg["prefix"] + os.path.basename(ckpt_path)
    s3.upload_file(ckpt_path, ckpt_cfg["bucket"], key)
    print(f"Checkpoint uploaded to s3://{ckpt_cfg['bucket']}/{key}")

def load_ckpt(ckpt_cfg, device):
    s3 = boto3.client("s3")
    bucket = ckpt_cfg["bucket"]
    key = ckpt_cfg["prefix"] + "latest.pt"

    try:
        with tempfile.NamedTemporaryFile() as tmp_file:
            print(f"Downloading checkpoint s3://{bucket}/{key}")
            s3.download_file(bucket, key, tmp_file.name)
            ckpt = torch.load(tmp_file.name, map_location=device)
            print(f"checkpoint loaded from S3 (epoch {ckpt['epoch'] + 1})")
            return ckpt["epoch"] + 1, ckpt["step"], ckpt
    except s3.exceptions.ClientError as e:
        print(f"No checkpoint found in s3://{bucket}/{key}, starting fresh")
        return 0, 0, None

def main():
    download_data()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    if "RANK" not in os.environ and "OMPI_COMM_WORLD_RANK" in os.environ:
        os.environ["RANK"] = os.environ["OMPI_COMM_WORLD_RANK"]
        os.environ["WORLD_SIZE"] = os.environ["OMPI_COMM_WORLD_SIZE"]
        os.environ["LOCAL_RANK"] = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")
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

    if rank == 0:
        print("SERVER STARTED")
        start_http_server(8001, addr="0.0.0.0")

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

    # Load checkpoint if existing
    ckpt_cfg = config["checkpoint"]
    os.makedirs(ckpt_cfg["dir"], exist_ok=True)
    start_epoch, global_step, ckpt = load_ckpt(ckpt_cfg, device)

    if ckpt:
        state_dict = { f"module.{k}" if not k.startswith("module.") else k : v for k, v in ckpt["model"].items() }
        ddp_model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optim"])

    dist.barrier()

    if rank == 0:
        start_time = time.time()

    for epoch in range(config["training"]["epochs"]):
        ddp_model.train()
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            sync_context = (
                contextlib.nullcontext()
                if (batch_idx + 1) % accum_steps == 0
                else ddp_model.no_sync()
            )
            with sync_context:
                output = ddp_model(x)
                loss = loss_fn(output, y) / accum_steps
                loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                global_step += 1
                grad_norm = 0
                
                if rank == 0 and global_step % config["wandb"]["log_interval"] == 0:
                    for p in ddp_model.parameters():
                        if p.grad is not None:
                            grad_norm += p.grad.data.norm(2).item() ** 2
                    grad_norm = grad_norm ** 0.5

                optimizer.step()
                optimizer.zero_grad()

                if rank == 0 and global_step % config["wandb"]["log_interval"] == 0:
                    print("Outputting loss gauge: ", loss.item() * accum_steps)
                    if torch.cuda.is_available():
                        mem_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
                        gpu_mem_usage.set(mem_mb)
                    loss_gauge.set(loss.item() * accum_steps)

                    wandb.log({
                        "loss": loss.item() * accum_steps,
                        "grad_norm": grad_norm,
                        "step": global_step,
                        "epoch": epoch
                    })
                    print(f"[Rank {rank}] Epoch {epoch} Step {global_step} Loss {loss.item() * accum_steps:.4f}")

                if rank == 0 and global_step % ckpt_cfg["save_interval"] == 0:
                    state = {
                        "epoch":   epoch,
                        "step":    global_step,
                        "model":   ddp_model.module.state_dict(),
                        "optim":   optimizer.state_dict(),
                        "config":  config,
                        "time":    datetime.datetime.utcnow().isoformat()
                    }
                    ckpt_name = f"ckpt_e{epoch}_s{global_step}.pt"
                    save_ckpt(state, os.path.join(ckpt_cfg["dir"], ckpt_name), ckpt_cfg)

                    # Use latest checkpoint for resume
                    save_ckpt(state, os.path.join(ckpt_cfg["dir"], "latest.pt"), ckpt_cfg)
                dist.barrier()

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
