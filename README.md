# BLS12-381 CUDA Backend

High-performance GPU-accelerated BLS12-381 cryptographic operations for zero-knowledge proof systems.

This backend provides CUDA-accelerated implementations of computationally intensive cryptographic primitives used in zkSNARK proof generation, offering significant performance improvements over CPU-only implementations for large-scale operations.

## Overview

The BLS12-381 CUDA Backend is a purpose-built acceleration layer for zero-knowledge proof systems that leverage the BLS12-381 elliptic curve. It implements critical operations that typically dominate prover runtime:

- **Multi-Scalar Multiplication (MSM)** - The core operation for polynomial commitments, computing `sum(scalars[i] * points[i])` on both G1 and G2 curves
- **Number Theoretic Transform (NTT)** - Fast polynomial multiplication and evaluation used throughout the proving process
- **Vector Operations** - Element-wise field arithmetic operations on large arrays

### Performance Characteristics

From the ICICLE benchmarks and real-world usage:

- **MSM**: Faster than CPU implementations for operations with 2^20 or more points
- **NTT**: Faster for large polynomial operations (2^22 elements)
- **Vector Operations**: Faster for element-wise operations on large arrays

The backend uses intelligent hybrid execution, automatically selecting GPU or CPU based on operation size to optimize both throughput and latency.

## Key Features

### Zero-Copy Type Conversions
Efficient memory layout compatibility between midnight-curves types and ICICLE types enables zero-copy conversions using `transmute`, eliminating unnecessary data copying between host and device memory.

### Asynchronous GPU Operations
Full support for CUDA streams enables:
- Overlapping computation and memory transfers
- Pipelined batch operations
- Non-blocking GPU execution for improved throughput

### Intelligent Device Selection
Automatic hybrid CPU/GPU execution based on operation size:
- Large operations (≥ 2^15 points) use GPU for maximum throughput
- Small operations use CPU (BLST) to avoid GPU overhead
- Configurable thresholds via environment variables

### Production-Ready Safety
- Compile-time assertions verify memory layout compatibility
- RAII stream management prevents resource leaks
- Comprehensive error handling with detailed diagnostics
- Memory tracking and leak detection in debug builds

## Prerequisites

- **NVIDIA GPU**: CUDA compute capability 7.0 or higher (Volta, Turing, Ampere, Ada architectures)
- **CUDA Toolkit**: Version 12.0 or later
- **CMake**: Version 3.18 or higher
- **C++ Compiler**: GCC 9+ or Clang 10+ with C++17 support
- **Rust**: Version 1.70 or later

## Building

### 1. Build the CUDA Backend

```bash
cd bls12-381
./build.sh
```

Or manually:

```bash
cd bls12-381
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make icicle -j$(nproc)
sudo make icicle-install
```

This installs the ICICLE CUDA backend to `/opt/icicle/lib/backend/`.

### 2. Run Tests

The backend includes comprehensive test suites:

```bash
# Run all tests
make test

# Run specific test suites
./build/test_msm_security
./build/test_ntt_security
./build/test_curve_operations
./build/test_field_properties
```

## Integration

For detailed integration instructions with midnight-zk or other zero-knowledge proof systems, please refer to the [Integration Guide](docs/integration.md).

## Configuration

Control device selection via environment variables:

- `MIDNIGHT_DEVICE`: Device selection mode
  - `auto` (default): GPU for large ops (≥ 2^15 points), CPU for small
  - `gpu`: Force GPU for all operations
  - `cpu`: Force CPU for all operations (disable GPU)

- `ICICLE_BACKEND_INSTALL_DIR`: Path to ICICLE backend library
  - Default: `/opt/icicle/lib/backend`

- `MIDNIGHT_GPU_MIN_K`: Minimum K value for GPU usage
  - Default: 15 (meaning 2^15 = 32768 points)

## Architecture

The backend is structured as a layered architecture separating high-level Rust APIs from low-level CUDA implementations:

### Layer Overview

```
┌─────────────────────────────────────────┐
│   Zero-Knowledge Proof System Layer     │  (midnight-zk or other)
├─────────────────────────────────────────┤
│   Rust API Layer (core/)                │  Type-safe, zero-copy conversions
├─────────────────────────────────────────┤
│   ICICLE Runtime Layer                  │  Backend loading, device management
├─────────────────────────────────────────┤
│   CUDA Implementation (bls12-381/)      │  Optimized kernels, memory management
├─────────────────────────────────────────┤
│   CUDA Driver & Hardware                │  NVIDIA GPU
└─────────────────────────────────────────┘
```

### Directory Structure

```
bls12-381-cuda-backend/
├── core/                      # Rust API layer
│   ├── mod.rs                # Public API surface and exports
│   ├── backend.rs            # ICICLE backend initialization and loading
│   ├── config.rs             # Device selection and configuration
│   ├── msm.rs                # MSM high-level API and orchestration
│   ├── ntt.rs                # NTT high-level API and orchestration
│   ├── stream.rs             # CUDA stream RAII wrapper
│   ├── types.rs              # Zero-copy type conversions
│   └── vecops.rs             # Vector operations API
│
├── bls12-381/                 # CUDA implementation layer
│   ├── include/              # C++ headers
│   │   ├── field.cuh         # Field arithmetic (Fq, Fr) with Montgomery form
│   │   ├── point.cuh         # Elliptic curve point operations (G1, G2)
│   │   ├── msm.cuh           # MSM algorithm (Pippenger)
│   │   ├── ntt.cuh           # NTT algorithm (Cooley-Tukey)
│   │   └── icicle_backend_api.cuh  # Backend registration interface
│   │
│   ├── src/
│   │   ├── backend/          # ICICLE backend integration
│   │   │   ├── icicle_field_api.cu    # Field operations registration
│   │   │   ├── icicle_curve_api.cu    # MSM implementation and registration
│   │   │   └── g2_registry.cu         # G2 curve operations registration
│   │   │
│   │   ├── curve/            # Curve operations
│   │   │   ├── msm_kernels.cu        # MSM CUDA kernels
│   │   │   └── point_ops.cu          # Point arithmetic kernels
│   │   │
│   │   ├── field/            # Field operations
│   │   │   ├── ntt_kernels.cu        # NTT CUDA kernels
│   │   │   └── vec_ops.cu            # Vector operation kernels
│   │   │
│   │   └── device/           # Device management
│   │       └── cuda_device_api.cu    # Memory and device utilities
│   │
│   ├── tests/                # Comprehensive test suites
│   │   ├── test_msm_security.cu      # MSM correctness and edge cases
│   │   ├── test_ntt_security.cu      # NTT correctness and edge cases
│   │   ├── test_curve_operations.cu  # Point operations validation
│   │   └── test_field_properties.cu  # Field arithmetic validation
│   │
│   └── CMakeLists.txt        # Build configuration
│
└── Cargo.toml                # Rust package manifest
```

### Design Principles

1. **Separation of Concerns**: Rust layer handles type safety and API design; CUDA layer focuses on performance
2. **Zero-Copy Operations**: Memory layouts are compatible to enable transmute-based conversions
3. **Fail-Fast Compilation**: Compile-time assertions catch layout mismatches before runtime
4. **Resource Safety**: RAII patterns ensure proper cleanup of GPU resources
5. **Extensibility**: Backend registration pattern allows easy addition of new operations

## Cargo Features

The backend supports the following Cargo features for build-time configuration:

- **`gpu`** - Enable GPU acceleration (default: disabled for compatibility)
  - When enabled, builds with ICICLE runtime and CUDA support
  - When disabled, provides stub implementations that always return errors
  
- **`trace-msm`** - Enable detailed MSM operation tracing
  - Logs MSM configuration, input sizes, and execution times
  - Useful for performance analysis and debugging
  
- **`trace-fft`** - Enable detailed NTT/FFT operation tracing
  - Logs NTT parameters, direction, and timing information
  - Helps identify bottlenecks in polynomial operations
  
- **`trace-all`** - Enable all tracing features
  - Equivalent to enabling both `trace-msm` and `trace-fft`
  - Recommended for comprehensive performance profiling

Example usage in `Cargo.toml`:

```toml
[dependencies]
midnight-bls12-381-cuda = { version = "0.1", features = ["gpu", "trace-all"] }
```

## Environment Variables

Runtime behavior can be controlled via environment variables:

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `MIDNIGHT_DEVICE` | `auto`, `gpu`, `cpu` | `auto` | Device selection strategy |
| `MIDNIGHT_GPU_MIN_K` | Integer (10-28) | `15` | Minimum log2(size) for GPU usage |
| `ICICLE_BACKEND_INSTALL_DIR` | Path | `/opt/icicle/lib/backend` | ICICLE backend library location |
| `RUST_LOG` | Log level | - | Enable tracing output (e.g., `RUST_LOG=debug`) |

### Device Selection Strategies

- **`auto`** (default): Intelligent hybrid execution
  - Operations with ≥ 2^K points use GPU
  - Smaller operations use CPU (BLST) to avoid GPU overhead
  - Balances throughput and latency
  
- **`gpu`**: Force all operations to GPU
  - Useful for benchmarking pure GPU performance
  - May hurt performance for small operations due to PCIe transfer overhead
  
- **`cpu`**: Disable GPU completely
  - All operations fall back to CPU (BLST)
  - Useful for compatibility testing or systems without GPUs

## Testing

The backend includes extensive test coverage across multiple dimensions:

### CUDA Test Suites

```bash
cd bls12-381/build

# Security and correctness tests
./test_msm_security          # MSM edge cases and known answer tests
./test_ntt_security          # NTT correctness and inverse property tests
./test_security_edge_cases   # Field and curve edge case validation

# Operations tests
./test_curve_operations      # Point addition, doubling, scalar multiplication
./test_field_properties      # Field arithmetic properties
./test_point_ops            # G1/G2 point operations
./test_vec_ops              # Vector operations validation

# Known answer tests
./test_known_answer_vectors  # Test vectors from reference implementations
```

### Rust Integration Tests

```bash
# Run Rust tests (requires GPU and built CUDA backend)
cargo test --features gpu

# Run with logging
RUST_LOG=debug cargo test --features gpu,trace-all
```

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
