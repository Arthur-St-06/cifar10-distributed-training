import os
import torch
import torchvision
import torchvision.transforms as transforms
import boto3
import botocore
import fcntl

def s3_object_exists(s3, bucket_name, s3_key):
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        else:
            raise

def download_and_save_cifar10(data_dir, bucket_name, s3_key, use_s3):
    if use_s3:
        s3 = boto3.client("s3")
    save_path = os.path.join(data_dir, s3_key)
    lock_path = os.path.join(data_dir, ".download.lock")

    with open(lock_path, "w") as lock_file:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX)

            if use_s3 and s3_object_exists(s3, bucket_name, s3_key):
                print(f"S3 object s3://{save_path} already exists. Skipping data download to s3 bucket.")
                use_s3 = False
                if os.path.isfile(save_path):
                    print(f"Dataset already saved. Skipping data download to local machine.")
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

            torch.save(train_dataset, save_path)
            print(f"CIFAR-10 training data saved to: {save_path}")

            if use_s3:
                s3.upload_file(save_path, bucket_name, s3_key)
                print(f"Uploaded {save_path} to s3://{bucket_name}/{s3_key}")
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)

def download_data(use_s3=True):
    dataset_path = os.getenv("DATA_PATH", "./data")
    bucket = os.getenv("S3_BUCKET", "arthur-cifar10-data")
    s3_key = os.getenv("S3_KEY", "cifar10_train.pt")

    download_and_save_cifar10(dataset_path, bucket, s3_key, use_s3)