FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python, OpenMPI, SSH, and other dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-venv \
    openmpi-bin openmpi-common libopenmpi-dev \
    openssh-client openssh-server \
    libgl1 pciutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Use Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

ENV PYTHONPATH=/app

# Set up SSH for MPIJob pod-to-pod communication
RUN mkdir -p /var/run/sshd && \
    echo "root:root" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#StrictModes.*/StrictModes no/' /etc/ssh/sshd_config

# Install Python dependencies
COPY src/requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY tests ./tests
RUN chmod +x src/entrypoint.sh

ENTRYPOINT ["./src/entrypoint.sh"]
