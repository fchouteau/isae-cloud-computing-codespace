#!/bin/bash
# train_remote.sh - Automated remote training on GCE
#
# This script automates the entire GCP training workflow:
#   1. Creates a Deep Learning VM
#   2. Copies the training script
#   3. Runs training in background
#   4. Waits for completion (polls GCS for results)
#   5. Deletes the VM
#   6. Displays download instructions
#
# Usage:
#   ./train_remote.sh          # Train for 5 epochs (default)
#   ./train_remote.sh 10       # Train for 10 epochs

set -e  # Exit on any error

# Configuration
GCS_BUCKET="gs://isae-sdd-de-2526"
RUN_ID="${USER}-$(date +%Y%m%d-%H%M%S)"
INSTANCE_NAME="training-vm-${RUN_ID}"
ZONE="europe-west1-b"
GCS_OUTPUT="${GCS_BUCKET}/runs/${RUN_ID}"
EPOCHS=${1:-5}  # Default 5 epochs, or pass as argument

echo "=============================================="
echo "GCP Remote Training Script"
echo "=============================================="
echo "Run ID:      ${RUN_ID}"
echo "Instance:    ${INSTANCE_NAME}"
echo "Zone:        ${ZONE}"
echo "Epochs:      ${EPOCHS}"
echo "Output:      ${GCS_OUTPUT}"
echo "=============================================="
echo ""

# Step 1: Create VM
echo "==> [1/6] Creating VM ${INSTANCE_NAME}..."
gcloud compute instances create ${INSTANCE_NAME} \
    --zone=${ZONE} \
    --image-family=pytorch-latest-cpu \
    --image-project=deeplearning-platform-release \
    --machine-type=n1-standard-2 \
    --scopes=storage-rw \
    --boot-disk-size=50GB

# Step 2: Wait for SSH to become available
echo "==> [2/6] Waiting for SSH to become available..."
until gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command="echo ready" 2>/dev/null; do
    echo "    Waiting for VM to be ready..."
    sleep 5
done
echo "    SSH is available!"

# Step 3: Copy training script
echo "==> [3/6] Copying training script..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gcloud compute scp "${SCRIPT_DIR}/train.py" ${INSTANCE_NAME}:~ --zone=${ZONE}

# Step 4: Start training in background
echo "==> [4/6] Starting training (${EPOCHS} epochs)..."
gcloud compute ssh ${INSTANCE_NAME} --zone=${ZONE} --command \
    "nohup python train.py --epochs ${EPOCHS} --output-gcs ${GCS_OUTPUT} > training.log 2>&1 &"
echo "    Training started in background on VM"

# Step 5: Poll for completion
echo "==> [5/6] Waiting for training to complete..."
echo "    (Checking GCS every 30 seconds for results)"
while ! gcloud storage ls ${GCS_OUTPUT}/metrics.json &> /dev/null; do
    echo "    $(date '+%H:%M:%S'): Training in progress..."
    sleep 30
done
echo "    Training complete! Results uploaded to GCS"

# Step 6: Delete VM
echo "==> [6/6] Deleting VM..."
gcloud compute instances delete ${INSTANCE_NAME} --zone=${ZONE} --quiet
echo "    VM deleted"

# Done
echo ""
echo "=============================================="
echo "Training Complete!"
echo "=============================================="
echo "Results available at: ${GCS_OUTPUT}"
echo ""
echo "To download results:"
echo "  mkdir -p ./results/${RUN_ID}"
echo "  gcloud storage cp ${GCS_OUTPUT}/* ./results/${RUN_ID}/"
echo ""
echo "Then open analyze.ipynb to visualize your results!"
echo "=============================================="
