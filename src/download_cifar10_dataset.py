import os
import torch
import torchvision
import torchvision.transforms as transforms

def s3_object_exists(s3, bucket_name, s3_key):
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise

def download_and_save_cifar10(data_dir, bucket_name, s3_key):
    s3 = boto3.client("s3")

    if s3_object_exists(s3, bucket_name, s3_key):
        print(f"S3 object s3://{bucket_name}/{s3_key} already exists. Skipping data download.")
        return

    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    save_path = os.path.join(data_dir, "cifar10_train.pt")
    torch.save(train_dataset, save_path)
    print(f"CIFAR-10 training data saved to: {save_path}")

    s3.upload_file(save_path, bucket_name, s3_key)
    print(f"Uploaded {save_path} to s3://{bucket_name}/{s3_key}")

if __name__ == "__main__":
    dataset_path = os.getenv("DATA_PATH", "./data")
    bucket = os.getenv("S3_BUCKET", "arthur-cifar10-data")
    s3_key = os.getenv("S3_KEY", "cifar10_train.pt")

    download_and_save_cifar10(dataset_path, bucket, s3_key)
