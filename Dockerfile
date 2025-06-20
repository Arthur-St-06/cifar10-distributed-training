FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

CMD ["torchrun", "--nproc-per-node=1", "--nnodes=2", "--node_rank=0", "--rdzv_backend=c10d", "--rdzv_endpoint=localhost:29500", "train.py"]