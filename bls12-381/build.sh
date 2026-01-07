#!/bin/bash
# Build script for CUDA backend
# This script builds the open-source CUDA backend libraries
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
INSTALL_DIR="${SCRIPT_DIR}/install"

# Parse arguments
BUILD_TYPE="Release"
CUDA_ARCH=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--debug] [--arch <CUDA_ARCH>] [--install-dir <DIR>]"
            exit 1
            ;;
    esac
done

echo "=== Building CUDA Backend for BLS12-381 ==="
echo "Build type: ${BUILD_TYPE}"
echo "Install dir: ${INSTALL_DIR}"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
echo "CUDA version: ${CUDA_VERSION}"

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
    "-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
)

if [ -n "${CUDA_ARCH}" ]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}")
fi

echo "Configuring..."
cmake "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

# Build
echo "Building..."
cmake --build . --parallel $(nproc)

# Install
echo "Installing..."
cmake --install .

echo ""
echo "=== Build Complete ==="
echo ""
echo "Libraries installed to: ${INSTALL_DIR}/lib/backend/bls12_381/cuda/"
echo ""
echo "To use as replacement for Icicle CUDA backend:"
echo ""
echo "Copy libraries to /opt/icicle/lib/backend/bls12_381/cuda/"
echo ""
