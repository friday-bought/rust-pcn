#!/bin/bash
# Download and extract CIFAR-10 binary version
set -e

DEST_DIR="${1:-/datasets}"
CIFAR_DIR="${DEST_DIR}/cifar-10"
URL="https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"

mkdir -p "${CIFAR_DIR}"
echo "Downloading CIFAR-10 from ${URL}..."
curl -fSL "${URL}" -o "${CIFAR_DIR}/cifar-10-binary.tar.gz"
echo "Extracting..."
cd "${CIFAR_DIR}"
tar xzf cifar-10-binary.tar.gz
rm -f cifar-10-binary.tar.gz
echo "CIFAR-10 extracted to ${CIFAR_DIR}/cifar-10-batches-bin/"
ls -la "${CIFAR_DIR}/cifar-10-batches-bin/"
