import torch
from torchvision import datasets, transforms

def save_dataset():
    transform = transforms.ToTensor()
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    torch.save(dataset, "./data/mnist_train.pt")

if __name__ == "__main__":
    save_dataset()
