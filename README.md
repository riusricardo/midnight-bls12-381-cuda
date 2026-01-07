# midnight-bls12-381-cuda

CUDA-accelerated BLS12-381 cryptographic operations for midnight-zk using ICICLE.

This crate provides GPU-accelerated implementations of computationally intensive operations:
- **Multi-Scalar Multiplication (MSM)** on G1 and G2 curves
- **Number Theoretic Transform (NTT)** for polynomial operations  
- **Vector Operations** for field arithmetic

## Features

- **Zero-copy type conversions** between midnight-curves and ICICLE types
- **Async GPU operations** with proper stream management
- **Automatic CPU fallback** for small operations
- **Production-ready** with compile-time safety assertions

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
midnight-bls12-381-cuda = { version = "0.1", features = ["gpu"] }
```

### Prerequisites

1. NVIDIA GPU with CUDA support
2. CUDA Toolkit 12.0 or later
3. CMake 3.18+

### Building the CUDA Backend

```bash
cd bls12-381
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make icicle -j$(nproc)
sudo make icicle-install
```

This installs the ICICLE CUDA backend to `/opt/icicle/lib/backend/`.

## Usage

```rust
use midnight_bls12_381_cuda::{
    ensure_backend_loaded, 
    should_use_gpu, 
    GpuMsmContext
};

// Initialize GPU backend (call once at startup)
ensure_backend_loaded()?;

// Check if GPU should be used for this operation size
if should_use_gpu(points.len()) {
    let ctx = GpuMsmContext::new()?;
    
    // Upload bases to GPU (done once, cached)
    let device_bases = ctx.upload_bases(&points)?;
    
    // Perform MSM on GPU
    let result = ctx.msm(&scalars, &device_bases)?;
} else {
    // Use CPU fallback (BLST) for small operations
    // ...
}
```

## Configuration

Control device selection via environment variables:

- `MIDNIGHT_DEVICE`: Device selection mode
  - `auto` (default): GPU for large ops (≥ 2^16 points), CPU for small
  - `gpu`: Force GPU for all operations
  - `cpu`: Force CPU for all operations (disable GPU)

- `ICICLE_BACKEND_INSTALL_DIR`: Path to ICICLE backend library
  - Default: `/opt/icicle/lib/backend`

- `MIDNIGHT_GPU_MIN_K`: Minimum K value for GPU usage
  - Default: 16 (meaning 2^16 = 65536 points)

## Architecture

This crate is designed as a standalone dependency that can be used in any Rust project requiring GPU-accelerated BLS12-381 operations:

```
midnight-bls12-381-cuda/
├── core/                 # Main library code
│   ├── mod.rs           # Public API and exports
│   ├── backend.rs       # ICICLE backend initialization
│   ├── config.rs        # Device configuration
│   ├── msm.rs           # Multi-scalar multiplication
│   ├── ntt.rs           # Number theoretic transform
│   ├── stream.rs        # GPU stream management
│   ├── types.rs         # Type conversions
│   └── vecops.rs        # Vector operations
├── bls12-381/           # CUDA backend C++ code
│   ├── src/             # CUDA implementations
│   └── CMakeLists.txt   # Build configuration
└── Cargo.toml           # Rust package manifest
```

## Features

- `gpu` - Enable GPU acceleration (default: disabled)
- `trace-msm` - Add MSM operation tracing
- `trace-fft` - Add FFT operation tracing  
- `trace-all` - Enable all tracing features

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
