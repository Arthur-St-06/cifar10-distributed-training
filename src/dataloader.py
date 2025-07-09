import os
import torch

def get_dataloader(batch_size, rank, world_size):
    dataset_path = os.getenv("DATA_PATH1", "./data")
    full_path = os.path.join(dataset_path, "cifar10_train.pt")

    dataset = torch.load(full_path)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader
