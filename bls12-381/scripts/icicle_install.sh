#!/bin/bash
# =============================================================================
# ICICLE Backend Installation Script
# =============================================================================
# This script backs up original ICICLE libraries and installs our open-source
# replacements.
#
# Usage: sudo ./scripts/icicle_install.sh <build_dir> [icicle_dir]
#
# Arguments:
#   build_dir  - Path to cmake build directory containing the .so files
#   icicle_dir - ICICLE installation directory (default: /opt/icicle)
# =============================================================================

set -e

BUILD_DIR="${1:-.}"
ICICLE_DIR="${2:-/opt/icicle}"

BLS12_381_DIR="${ICICLE_DIR}/lib/backend/bls12_381/cuda"
CUDA_DIR="${ICICLE_DIR}/lib/backend/cuda"

echo "[INSTALL] Installing ICICLE backends..."

# Check directories exist
if [ ! -d "${BLS12_381_DIR}" ]; then
    echo "[ERROR] ICICLE not found at ${BLS12_381_DIR}"
    echo "        Please install ICICLE first."
    exit 1
fi

if [ ! -d "${CUDA_DIR}" ]; then
    echo "[ERROR] ICICLE device backend not found at ${CUDA_DIR}"
    exit 1
fi

# Check source libraries exist
if [ ! -f "${BUILD_DIR}/libicicle_backend_cuda_field_bls12_381.so" ]; then
    echo "[ERROR] Field library not found at ${BUILD_DIR}/libicicle_backend_cuda_field_bls12_381.so"
    echo "        Please build first: make icicle"
    exit 1
fi

if [ ! -f "${BUILD_DIR}/libicicle_backend_cuda_curve_bls12_381.so" ]; then
    echo "[ERROR] Curve library not found at ${BUILD_DIR}/libicicle_backend_cuda_curve_bls12_381.so"
    echo "        Please build first: make icicle"
    exit 1
fi

if [ ! -f "${BUILD_DIR}/libicicle_backend_cuda_device.so" ]; then
    echo "[ERROR] Device library not found at ${BUILD_DIR}/libicicle_backend_cuda_device.so"
    echo "        Please build first: make icicle"
    exit 1
fi

# Backup original libraries (only if .orig doesn't exist)
echo "[BACKUP] Backing up original libraries..."

if [ ! -f "${BLS12_381_DIR}/libicicle_backend_cuda_field_bls12_381.so.orig" ]; then
    if [ -f "${BLS12_381_DIR}/libicicle_backend_cuda_field_bls12_381.so" ]; then
        cp "${BLS12_381_DIR}/libicicle_backend_cuda_field_bls12_381.so" \
           "${BLS12_381_DIR}/libicicle_backend_cuda_field_bls12_381.so.orig"
        echo "  - Backed up field library"
    fi
fi

if [ ! -f "${BLS12_381_DIR}/libicicle_backend_cuda_curve_bls12_381.so.orig" ]; then
    if [ -f "${BLS12_381_DIR}/libicicle_backend_cuda_curve_bls12_381.so" ]; then
        cp "${BLS12_381_DIR}/libicicle_backend_cuda_curve_bls12_381.so" \
           "${BLS12_381_DIR}/libicicle_backend_cuda_curve_bls12_381.so.orig"
        echo "  - Backed up curve library"
    fi
fi

if [ ! -f "${CUDA_DIR}/libicicle_backend_cuda_device.so.orig" ]; then
    if [ -f "${CUDA_DIR}/libicicle_backend_cuda_device.so" ]; then
        cp "${CUDA_DIR}/libicicle_backend_cuda_device.so" \
           "${CUDA_DIR}/libicicle_backend_cuda_device.so.orig"
        echo "  - Backed up device library"
    fi
fi

# Install new libraries
echo "[INSTALL] Copying new libraries..."

cp "${BUILD_DIR}/libicicle_backend_cuda_field_bls12_381.so" "${BLS12_381_DIR}/"
echo "  - Installed field library"

cp "${BUILD_DIR}/libicicle_backend_cuda_curve_bls12_381.so" "${BLS12_381_DIR}/"
echo "  - Installed curve library"

cp "${BUILD_DIR}/libicicle_backend_cuda_device.so" "${CUDA_DIR}/"
echo "  - Installed device library"

echo ""
echo "[OK] ICICLE backends installed!"
echo ""
echo "To restore original libraries:"
echo "  sudo cp ${ICICLE_DIR}/lib/backend/bls12_381/cuda/*.orig ${ICICLE_DIR}/lib/backend/bls12_381/cuda/"
echo "  sudo cp ${ICICLE_DIR}/lib/backend/cuda/*.orig ${ICICLE_DIR}/lib/backend/cuda/"
