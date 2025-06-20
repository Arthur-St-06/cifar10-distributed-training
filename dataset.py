from torchvision import datasets, transforms

def get_dataloader(batch_size, rank, world_size):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader
