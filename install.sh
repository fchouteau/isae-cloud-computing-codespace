#!/bin/bash
set -euo pipefail

# Add Google Cloud SDK repo (modern keyring approach)
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /etc/apt/keyrings/cloud.google.gpg
echo "deb [signed-by=/etc/apt/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list

# Install packages
sudo apt update
sudo apt install -y --no-install-recommends google-cloud-cli ffmpeg libsm6 libxext6

# Cleanup apt cache
sudo rm -rf /var/lib/apt/lists/*
