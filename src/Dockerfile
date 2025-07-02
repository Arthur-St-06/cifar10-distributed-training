FROM python:3.10

WORKDIR /app

# Install OpenMPI, SSH, and other dependencies
RUN apt-get update && apt-get install -y \
    openmpi-bin openmpi-common libopenmpi-dev \
    openssh-client openssh-server \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Configure SSH (required for MPIJob pod communication)
RUN mkdir -p /var/run/sshd && \
    echo "root:root" | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN sed -i 's/#StrictModes.*/StrictModes no/' /etc/ssh/sshd_config

ENV PYTHONPATH=/app

# Install Python dependencies
COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir src
COPY src/train.py src/
COPY src/model.py src/
COPY src/dataloader.py src/
COPY src/entrypoint.sh src/
COPY tests ./tests
RUN chmod +x src/entrypoint.sh

ENTRYPOINT ["./src/entrypoint.sh"]
