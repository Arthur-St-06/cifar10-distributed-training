import os
import torch
import boto3
import botocore

def download_from_s3_if_missing(local_path, bucket_name, s3_key):
    if os.path.exists(local_path):
        print(f"Found local dataset at {local_path}, skipping download.")
        return

    print(f"Downloading from s3://{bucket_name}/{s3_key} to {local_path}...")
    s3 = boto3.client("s3")
    os.makedirs(os.path.dirname(local_path))
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        print("Download complete.")
    except botocore.exceptions.ClientError as e:
        print(f"Failed to download: {e}")
        raise

def get_dataloader(batch_size, rank, world_size):
    dataset_path = os.getenv("DATA_PATH", "/app/src/data")
    full_path = os.path.join(dataset_path, "cifar10_train.pt")

    bucket_name = os.getenv("S3_BUCKET", "arthur-cifar10-data")
    s3_key = os.getenv("S3_KEY", "cifar10_train.pt")

    download_from_s3_if_missing(full_path, bucket_name, s3_key)

    dataset = torch.load(full_path)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader
