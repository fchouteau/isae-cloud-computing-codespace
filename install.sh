#!/bin/bash
set -euo pipefail

# Install Google Cloud CLI using manual installation with curl
# Reference: https://cloud.google.com/sdk/docs/install-sdk#linux

INSTALL_DIR="/opt/google-cloud-sdk"
GCLOUD_VERSION="google-cloud-cli-linux-x86_64.tar.gz"

# Check if gcloud is already installed
if command -v gcloud &> /dev/null; then
    echo "Google Cloud CLI is already installed: $(gcloud --version | head -n 1)"
    echo "Skipping installation."
    exit 0
fi

# Also check if installation directory exists
if [[ -d "${INSTALL_DIR}" ]]; then
    echo "Google Cloud CLI installation directory already exists at ${INSTALL_DIR}"
    echo "Skipping installation."
    exit 0
fi

# Download Google Cloud CLI
echo "Downloading Google Cloud CLI..."
curl -fsSL -o /tmp/${GCLOUD_VERSION} \
    https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/${GCLOUD_VERSION}

# Extract to /opt
echo "Extracting Google Cloud CLI to ${INSTALL_DIR}..."
sudo mkdir -p /opt
sudo tar -xzf /tmp/${GCLOUD_VERSION} -C /opt

# Run install script non-interactively (updates PATH in shell profiles)
echo "Running Google Cloud CLI install script..."
sudo ${INSTALL_DIR}/install.sh --quiet --path-update true

# Add gcloud to PATH for all users via /etc/profile.d
echo "Setting up environment variables..."
echo 'export PATH="/opt/google-cloud-sdk/bin:$PATH"' | sudo tee /etc/profile.d/gcloud.sh
sudo chmod +x /etc/profile.d/gcloud.sh

# Also add to .bashrc for the current user (for immediate availability)
if ! grep -q "google-cloud-sdk/bin" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="/opt/google-cloud-sdk/bin:$PATH"' >> ~/.bashrc
fi

# Cleanup
rm -f /tmp/${GCLOUD_VERSION}

echo "Google Cloud CLI installed successfully!"
echo "Run 'source ~/.bashrc' or open a new terminal to use gcloud."
