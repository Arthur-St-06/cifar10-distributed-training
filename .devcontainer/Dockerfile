FROM nvidia/cuda:12.1.0-base-ubuntu22.04

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip curl tzdata

RUN pip install --upgrade pip
RUN pip install debugpy

CMD ["sleep", "infinity"]
