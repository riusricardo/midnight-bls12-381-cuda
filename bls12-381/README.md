# BLS12-381 CUDA Backend

High-performance CUDA implementation of BLS12-381 cryptographic operations, providing GPU-accelerated field and elliptic curve arithmetic for zero-knowledge proof systems.

## Overview

This library provides open-source CUDA backend implementations compatible with the [ICICLE](https://github.com/ingonyama-zk/icicle) cryptographic library interface. It includes optimized GPU kernels for:

- **Field Operations**: BLS12-381 scalar field (Fr) arithmetic including NTT/iNTT and vector operations
- **Curve Operations**: G1 and G2 elliptic curve point operations including Multi-Scalar Multiplication (MSM)
- **Device Management**: CUDA device API for memory management and stream handling

## Features

- **High Performance**: Optimized CUDA kernels for cryptographic operations
- **ICICLE Compatible**: Drop-in replacement for ICICLE's CUDA backend libraries
- **Full BLS12-381 Support**: Both G1 and G2 curve support
- **Multi-GPU Architecture Support**: Builds for Turing through Blackwell GPUs (RTX 20xx to RTX 50xx, A100, H100)
- **Static Linking Option**: Redistributable binaries with static CUDA runtime

## Build Requirements

- CUDA Toolkit 11.0+ (12.x recommended)
- CMake 3.18+
- C++17 compatible compiler

## Building

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `CUDA_STATIC_RUNTIME` | ON | Use static CUDA runtime for redistributability |
| `BUILD_TESTS` | ON | Build test executables |
| `G2_ENABLED` | ON | Enable G2 curve support |
| `MULTI_GPU_ARCH` | ON | Build for multiple GPU architectures |
| `GLV_ENABLED` | OFF | Enable GLV endomorphism (experimental) |

### Example: Development Build (Faster)

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCUDA_STATIC_RUNTIME=OFF \
         -DMULTI_GPU_ARCH=OFF
```

## Output Libraries

The build produces three shared libraries:

| Library | Description |
|---------|-------------|
| `libicicle_backend_cuda_field_bls12_381.so` | Field operations (NTT, vector ops) |
| `libicicle_backend_cuda_curve_bls12_381.so` | Curve operations (MSM, point ops) |
| `libicicle_backend_cuda_device.so` | Device API (memory, streams) |

## Installation

To install to the ICICLE directory (default `/opt/icicle`):

```bash
sudo make icicle-install
```

Or specify a custom location:

```bash
cmake .. -DICICLE_INSTALL_DIR=/path/to/icicle
make icicle-install
```

## Project Structure

```
cuda-backend/
├── CMakeLists.txt          # Build configuration
├── include/                # Header files
│   ├── field/              # Field arithmetic headers
│   ├── curve/              # Curve operation headers
│   └── common/             # Shared utilities
├── src/
│   ├── field/              # Field implementations
│   │   ├── ntt_kernels.cu      # NTT (Number Theoretic Transform)
│   │   └── vec_ops.cu          # Vector operations
│   ├── curve/              # Curve implementations
│   │   ├── msm_kernels.cu      # Multi-Scalar Multiplication
│   │   └── point_ops.cu        # Point arithmetic
│   ├── device/             # Device management
│   └── backend/            # ICICLE API integration
├── tests/                  # Test suite
└── scripts/                # Build and utility scripts
```

## Supported Operations

### Field Operations (Fr)
- Modular arithmetic (add, sub, mul, inv)
- Number Theoretic Transform (NTT/iNTT)
- Vector operations (add, sub, mul, element-wise)
- Montgomery form conversions

### Curve Operations (G1/G2)
- Point addition and doubling
- Scalar multiplication
- Multi-Scalar Multiplication (MSM) using Pippenger's algorithm
- Affine ↔ Projective conversions
- Batch operations

## Performance

MSM performance scales with input size. For large inputs (2^20+ points), GPU acceleration provides significant speedup over CPU implementations.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](../LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.
