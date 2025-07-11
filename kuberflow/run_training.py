# sudo chown -R $USER:$USER ~/kube-download
# kubectl port-forward -n monitoring svc/prometheus-server 9090:80

from jinja2 import Template
import subprocess
import os
import yaml

def submit_training_job(
    image,
    job_name = None,
    script = "src/train.py",
    num_workers = 2,
    dataset_path = "/mnt/data",
    num_gpus = 0,
    eksctl_name = "main-cluster",
    eksctl_version = "1.30",
    eksctl_region = "us-west-2",
    eksctl_nodegroup_name = "linux-nodes",
    eksctl_node_type = "t3.medium",
    eksctl_nodes = "2",
    eksctl_setup = True,
    env_setup = True
):
    if eksctl_setup:
        print("Creating EKS cluster...")
        subprocess.run([
            "eksctl", "create", "cluster",
            "--name", eksctl_name,
            "--version", eksctl_version,
            "--region", eksctl_region,
            "--nodegroup-name", eksctl_nodegroup_name,
            "--node-type", eksctl_node_type,
            "--nodes", eksctl_nodes
        ], check=True)

        print("Getting IAM role name from CloudFormation...")
        role_name = subprocess.check_output([
            "aws", "cloudformation", "list-stack-resources",
            "--stack-name", f"eksctl-{eksctl_name}-nodegroup-{eksctl_nodegroup_name}",
            "--region", eksctl_region,
            "--query", "StackResourceSummaries[?ResourceType=='AWS::IAM::Role'].PhysicalResourceId",
            "--output", "text"
        ], text=True).strip()

        print(f"Attaching AmazonS3FullAccess policy to role: {role_name}...")
        subprocess.run([
            "aws", "iam", "attach-role-policy",
            "--role-name", role_name,
            "--policy-arn", "arn:aws:iam::aws:policy/AmazonS3FullAccess"
        ], check=True)

    if env_setup:
        # Create pv and pvc yaml configs
        with open("pv-pvc-template.yaml.j2") as f:
            pv_pvc_template = Template(f.read())

        pv_pvc_rendered_yaml = pv_pvc_template.render(
            dataset_path=dataset_path
        )

        pv_pvc_yaml_path = f"/tmp/pv-pvc.yaml"
        with open(pv_pvc_yaml_path, "w") as f:
            f.write(pv_pvc_rendered_yaml)

        #Run setup commands
        #print("Starting Minikube...")
        #subprocess.run(["minikube", "start", "--gpus=all"], check=True)

        #print("Creating /mnt/data in Minikube...")
        #subprocess.run(["minikube", "ssh", "--", "sudo", "mkdir", "-p", "/mnt/data"], check=True)

        # TODO Add s3 support
        #print("Copying dataset into Minikube...")
        #subprocess.run(["minikube", "cp", "data/cifar10_train.pt", "/mnt/data/cifar10_train.pt"], check=True)

        # print("Mounting fsx for lustre...")
        # subprocess.run(["kubectl", "apply", "-f", "fsx-mount.yaml"], check=True)

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

        print("Installing prometheus...")
        subprocess.run(["helm", "repo", "add", "prometheus-community", "https://prometheus-community.github.io/helm-charts"], check=True)
        subprocess.run(["helm", "repo", "update"], check=True)
        subprocess.run(["helm", "install", "prometheus", "prometheus-community/kube-prometheus-stack"], check=True)

        print("Applying Prometheus metrics service...")
        subprocess.run(["kubectl", "apply", "-f", "metrics-service.yaml"], check=True)

        print("Applying main service monitor...")
        subprocess.run(["kubectl", "apply", "-f", "main-service-monitor.yaml"], check=True)

        print("Installing DCGM exporter...")
        subprocess.run(["helm", "repo", "add", "gpu-helm-charts", "https://nvidia.github.io/dcgm-exporter/helm-charts"], check=True)

        subprocess.run(["helm", "repo", "update"], check=True)

        subprocess.run(["helm", "install", "--generate-name", "gpu-helm-charts/dcgm-exporter"], check=True)

        print("Applying nvidia service monitor...")
        subprocess.run(["kubectl", "apply", "-f", "nvidia-service-monitor.yaml"], check=True)

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
    subprocess.run("kubectl create configmap job-config --from-file=config.yaml --dry-run=client -o yaml | kubectl apply -f -", shell=True, check=True)

    print(f"Submitting training job {job_name}...")
    subprocess.run(["kubectl", "apply", "-f", job_yaml_path], check=True)

    return job_name

import argparse

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-setup", action="store_true", help="Run environment setup (Minikube, PVC, MPI operator, etc.)")
    parser.add_argument("--eksctl-setup", action="store_true", help="Run eksctl setup")
    args = parser.parse_args()

    config = load_config()

    job_config = config["training-setup"]
    eksctl_config = config["eksctl"]

    submit_training_job(
        image = job_config["image"],
        script = job_config["script"],
        num_workers = job_config["num_workers"],
        num_gpus = job_config["num_gpus"],
        eksctl_name = eksctl_config["name"],
        eksctl_version = eksctl_config["version"],
        eksctl_region = eksctl_config["region"],
        eksctl_nodegroup_name = eksctl_config["nodegroup-name"],
        eksctl_node_type = eksctl_config["node-type"],
        eksctl_nodes = eksctl_config["nodes"],
        eksctl_setup = args.eksctl_setup,
        env_setup = args.env_setup,
    )