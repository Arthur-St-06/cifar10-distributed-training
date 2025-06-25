from jinja2 import Template
import subprocess
import os
import yaml

def submit_training_job(
    image,
    job_name = None,
    script = "train.py",
    num_workers = 2,
    dataset_path = "/mnt/data",
    full_setup=True,
    num_gpus=0
):
    if full_setup:
        # Create pv and pvc yaml configs
        with open("pv-pvc-template.yaml.j2") as f:
            pv_pvc_template = Template(f.read())

        pv_pvc_rendered_yaml = pv_pvc_template.render(
            dataset_path=dataset_path,
        )

        pv_pvc_yaml_path = f"/tmp/pv-pvc.yaml"
        with open(pv_pvc_yaml_path, "w") as f:
            f.write(pv_pvc_rendered_yaml)

        # Run setup commands
        print("Starting Minikube...")
        subprocess.run(["minikube", "start", "--gpus=all"], check=True)

        print("Creating /mnt/data in Minikube...")
        subprocess.run(["minikube", "ssh", "--", "sudo", "mkdir", "-p", "/mnt/data"], check=True)

        print("Copying dataset into Minikube...")
        subprocess.run(["minikube", "cp", "/home/arthur/kube-download/mnist-detection-k8s/kuberflow/data/cifar10_train.pt", "/mnt/data/cifar10_train.pt"], check=True)

        #print("Creating configmap...")
        #subprocess.run(["kubectl", "create", "configmap", "job-config", "--from-file=../config.yaml"], check=True)

        if not os.path.exists("mpi-operator"):
            print("Cloning MPI Operator...")
            subprocess.run(["git", "clone", "https://github.com/kubeflow/mpi-operator"], check=True)

        print("Setting up MPI Operator...")
        subprocess.run(["git", "-C", "mpi-operator", "checkout", "v0.4.0"], check=True)
        subprocess.run(["kubectl", "apply", "-f", "mpi-operator/deploy/v2beta1/mpi-operator.yaml"], check=True)

        print("Applying PV and PVC YAMLs...")
        subprocess.run(["kubectl", "apply", "-f", pv_pvc_yaml_path], check=True)

        print("Applying wandb secret YAML...")
        subprocess.run(["kubectl", "apply", "-f", "wandb-secret.yaml"], check=True)

    # Create job yaml config
    if job_name is None:
        job_name = "mpi"

    with open("mpi-job-template.yaml.j2") as f:
        job_template = Template(f.read())

    job_rendered_yaml = job_template.render(
        job_name=job_name,
        image=image,
        script=script,
        num_workers=num_workers,
        num_gpus=num_gpus
    )

    job_yaml_path = f"/tmp/{job_name}.yaml"
    with open(job_yaml_path, "w") as f:
        f.write(job_rendered_yaml)

    print("Updating config...")
    subprocess.run("kubectl create configmap job-config --from-file=../config.yaml --dry-run=client -o yaml | kubectl apply -f -", shell=True, check=True)

    print(f"Submitting training job {job_name}...")
    subprocess.run(["kubectl", "apply", "-f", job_yaml_path], check=True)

    return job_name

import argparse

def load_job_config(path="../config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config["training_setup"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-setup", action="store_true", help="Run full environment setup (Minikube, PVC, MPI operator, etc.)")
    args = parser.parse_args()

    job_config = load_job_config()

    submit_training_job(
        image=job_config["image"],
        script=job_config["script"],
        num_workers=job_config["num_workers"],
        num_gpus=job_config["num_gpus"],
        full_setup=args.full_setup
    )