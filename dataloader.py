import torch

def get_dataloader(batch_size, rank, world_size):
    dataset = torch.load("./data/mnist_train.pt")

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader
