FROM python:3.10

# Set working directory
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

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your training code
COPY src/requirements.txt .
COPY src/train.py .
COPY src/model.py .
COPY src/dataloader.py .
COPY src/entrypoint.sh .
COPY tests ./tests
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
