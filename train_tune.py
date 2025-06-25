import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import SimpleModel
from tuner_dataloader import get_dataloader
from ray import tune, train
import yaml

def train_model(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    model = SimpleModel().to(device)

    dataloader = get_dataloader(batch_size=config["batch_size"])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    if config["use_wandb"]:
        import wandb
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="mnist", config={"lr": config["lr"]})

    accum_steps = config["accumulation_steps"]
    global_step = 0

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output, y) / accum_steps
            loss.backward()

            if (batch_idx + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                total_loss += loss.item() * accum_steps

        avg_loss = total_loss / len(dataloader)

        if config["use_wandb"]:
            wandb.log({"loss": avg_loss,
                        "step": global_step,
                        "epoch": epoch})
            print(f"[Epoch {epoch}] Loss {avg_loss}")
        
        train.report({"loss": total_loss / len(dataloader)})

    if config["use_wandb"]:
        wandb.finish()

    print("Training complete.")

def tune_model():
    with open("./config.yaml", "r") as f:
        base_config = yaml.safe_load(f)

    search_space = {
        "lr": tune.grid_search([1e-2, 1e-3, 1e-4, 5e-4]),
        "batch_size": tune.grid_search([16, 32, 64]),
        "accumulation_steps": base_config["training"]["accumulation_steps"],
        "epochs": base_config["training"]["epochs"],
        "log_interval": base_config["wandb"]["log_interval"],
        "device": base_config["training"]["device"],
        "use_wandb": base_config["wandb"]["use"]
    }

    tune.run(
        train_model,
        resources_per_trial={"cpu": 1, "gpu": 0},
        config=search_space,
        metric="loss",
        mode="min",
        name="tune_mnist"
    )

if __name__ == "__main__":
    tune_model()
