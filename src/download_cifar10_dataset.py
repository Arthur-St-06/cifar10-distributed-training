import os
import torch
import torchvision
import torchvision.transforms as transforms

def download_and_save_cifar10(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Download training data
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    save_path = os.path.join(data_dir, "cifar10_train.pt")
    torch.save(train_dataset, save_path)
    print(f"CIFAR-10 training data saved to: {save_path}")

if __name__ == "__main__":
    dataset_path = os.getenv("DATA_PATH", "./data")
    download_and_save_cifar10(dataset_path)
