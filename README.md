# CIFAR-10 Distributed Training with Kubernetes and PyTorch

A distributed, scalable training pipeline for CIFAR-10 image classification using PyTorch Distributed Data Parallel (DDP), Kubeflow MPI jobs on Kubernetes (EKS or local), monitoring via Prometheus and NVIDIA DCGM and experiment tracking via Weights & Biases.

## Project Structure

```
├── .devcontainer               # VSCode container config
│   ├── Dockerfile
│   └── devcontainer.json
├── .github
│   └── workflows
│       └── docker-publish.yml  # CI: build, test, push image
├── kuberflow                   # Kubernetes/MPI job configs & helpers
│   ├── config.yaml             # Main configuration
│   ├── main-service-monitor.yaml
│   ├── metrics-service.yaml
│   ├── mpi-job-template.yaml.j2
│   ├── nvidia-service-monitor.yaml
│   └── run_training.py         # EKS / env setup & job submit script
├── src                         # Training code and data utilities
│   ├── dataloader.py
│   ├── download_cifar10_dataset.py
│   ├── entrypoint.sh
│   ├── model.py
│   ├── requirements.txt
│   └── train.py
├── tests                       # Unit tests
|   ├── test_dataloader.py
|   └── test_model.py
├── .gitignore
└── Dockerfile                  # Production Docker image
```

## Prerequisites

* Docker & NVIDIA Drivers
* Python 3.10
* `kubectl`, `helm`, `eksctl`, `aws` CLI configured with IAM credentials
* Kubernetes cluster (Minikube or AWS EKS)

## Setup and Requirements

### Docker Image

* Built from `nvidia/cuda:12.2.0-runtime-ubuntu22.04`
* Configured for distributed training with OpenMPI and PyTorch
* SSH enabled for MPI communications

### Dependencies

See `src/requirements.txt`.

## CI/CD Workflow

The GitHub Actions workflow (`docker-publish.yml`) runs when pushing to main branch and automatically:

* Builds Docker images
* Runs unit tests inside Docker containers
* Pushes verified images to Docker Hub

Ensure the following GitHub Secrets are set:

* `DOCKERHUB_USERNAME`
* `DOCKERHUB_TOKEN`

## Training Setup with Kubeflow MPI Operator

### Kubernetes Configuration

* Cluster setup via `eksctl` defined in `kuberflow/config.yaml`
* ServiceMonitors and Prometheus integration for metrics collection
* Weights & Biases API key must be provided using the Kubernetes Secret (`kuberflow/wandb-secret.yaml`)

### Launching Training Jobs

The script `kuberflow/run_training.py`:

* Sets up EKS cluster, IAM roles, and S3 permissions
* Deploys Kubernetes manifests (MPIJobs, Prometheus, and monitoring tools)
* Starts distributed PyTorch training via MPI

## PyTorch Model

* Simple CNN model defined in `src/model.py`
* Distributed training script (`src/train.py`) includes checkpointing, logging, and metrics collection

## Data Handling

* CIFAR-10 dataset downloaded and optionally uploaded to Amazon S3 (`src/download_cifar10_dataset.py`)
* Distributed data loading via PyTorch Distributed Sampler (`src/dataloader.py`)

## Testing

* Unit tests (`tests/test_dataloader.py` and `tests/test_model.py`) verify data loading and model behavior

## Monitoring and Logging

* Prometheus metrics exported for GPU usage and training loss
* Weights & Biases integration for tracking experiments

To access Grafana for visual monitoring:

1. Port forward Grafana service:

```bash
kubectl port-forward svc/prometheus-grafana 3000:80
```

2. Log in with default credentials:

```
Username: admin
Password: prom-operator
```

3. Set up NVIDIA metrics dashboard:

* Select **Metrics**, click the **+** sign in the top right corner, select **Import**.
* Paste `12239` into the ID field, click **Load**.
* Choose **Prometheus** as data source and click **Import**.

### Local Development

Use VSCode with the provided devcontainer configuration for a reproducible development environment.

Set your Weights & Biases API key in `.devcontainer/devcontainer.json`

## Local Deployment with Minikube

To deploy locally with GPU support:

1. Start Minikube with GPU access:
   
```bash
minikube start --gpus=all
```

2. Set up environment and start training:

```bash
cd kuberflow/
python3 run\_training.py --env-setup
```

## Deploying on aws cluster with eksctl
To deploy the Kubernetes environment and start training:

```bash
cd kuberflow/
python3 run_training.py --eksctl-setup --env-setup
```
If environment is already set up, just do:
```bash
python3 run_training.py
```
