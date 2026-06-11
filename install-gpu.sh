#!/bin/bash
# Installs the NVIDIA Container Toolkit so Docker can use the GPU.
# Run once on the host:  sudo ./install-gpu.sh
set -e

if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo:  sudo ./install-gpu.sh"
    exit 1
fi

# 1. Add the NVIDIA Container Toolkit apt repo (Ubuntu).
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor --batch --yes --no-tty -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install.
apt-get update
apt-get install -y nvidia-container-toolkit

# 3. Wire it into Docker and restart the daemon.
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo
echo "Done. Verify with:"
echo "  docker run --rm --gpus all nvcr.io/nvidia/tritonserver:25.04-py3 nvidia-smi"
