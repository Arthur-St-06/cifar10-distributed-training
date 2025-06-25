import os
import torch
from torch.utils.data import DataLoader

def get_dataloader(batch_size):
    dataset_path = os.getenv("DATA_PATH")
    full_path = os.path.join(dataset_path, "cifar10_train.pt")
    dataset = torch.load(full_path)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader