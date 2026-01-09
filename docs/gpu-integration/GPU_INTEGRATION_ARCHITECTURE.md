# GPU Integration Architecture - Midnight ZK

## Overview

 The integration adds CUDA-based GPU acceleration for Multi-Scalar Multiplication (MSM) and Number Theoretic Transform (NTT/FFT) operations, which are the computational bottlenecks in zero-knowledge proof generation.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              midnight-proofs                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐           │
│  │   PLONK Prover   │    │    KZG Module    │    │   Domain/FFT     │           │
│  │   (prover.rs)    │───▶│   (kzg/mod.rs)   │───▶│  (domain.rs)     │           │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘           │
│           │                       │                       │                      │
│           ▼                       ▼                       ▼                      │
│  ┌────────────────────────────────────────────────────────────────────┐         │
│  │                       gpu_accel.rs (Bridge Module)                 │         │
│  │  • Single import point for all GPU functionality                   │         │
│  │  • Type-safe dispatch helpers (is_g1_affine, try_as_fq_slice)     │         │
│  │  • GPU availability checks (should_use_gpu, should_use_gpu_ntt)   │         │
│  │  • Re-exports from midnight-bls12-381-cuda                         │         │
│  └────────────────────────────────────────────────────────────────────┘         │
│                                    │                                             │
├────────────────────────────────────┼─────────────────────────────────────────────┤
│                                    ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐         │
│  │               midnight-bls12-381-cuda (External Crate)             │         │
│  │  • GpuMsmContext: MSM operations with device bases                 │         │
│  │  • PrecomputedBases: GPU-resident SRS bases                        │         │
│  │  • TypeConverter: Safe type conversions for ICICLE                 │         │
│  │  • NTT operations: forward/inverse NTT for Fq                      │         │
│  └────────────────────────────────────────────────────────────────────┘         │
│                                    │                                             │
│                                    ▼                                             │
│  ┌────────────────────────────────────────────────────────────────────┐         │
│  │                   ICICLE CUDA Backend (External)                   │         │
│  │  • GPU-accelerated MSM kernels                                     │         │
│  │  • GPU-accelerated NTT kernels                                     │         │
│  │  • Device memory management                                        │         │
│  └────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Single Import Point Pattern

All GPU functionality is accessed through `gpu_accel.rs`. Other modules should **never** import directly from `midnight_bls12_381_cuda` or `icicle_*`.

```rust
// Correct: Import from gpu_accel
use crate::gpu_accel::{should_use_gpu, PrecomputedBases, try_as_fq_slice};
```

### 2. Type-Safe Dispatch

The integration uses type-checking helpers to safely route computations to GPU:

```rust
// Type detection helpers
is_g1_affine::<C>()        // Check if C is midnight_curves::G1Affine
is_fq::<F>()               // Check if F is midnight_curves::Fq
try_as_fq_slice(coeffs)    // Safely reinterpret &[F] as &[Fq]

// High-level dispatch
dispatch_msm::<C>(coeffs, gpu_fn, cpu_fn)      // Auto-route MSM
dispatch_ntt_inplace(data, gpu_fn, cpu_fn)     // Auto-route NTT
```

### 3. Lazy GPU Base Caching

SRS (Structured Reference String) bases are uploaded to GPU memory lazily and cached:

```rust
// In ParamsKZG:
pub(crate) g_gpu: Arc<OnceCell<PrecomputedBases>>,           // Coefficient bases
pub(crate) g_lagrange_gpu: Arc<OnceCell<PrecomputedBases>>,  // Lagrange bases

// First access uploads bases; subsequent accesses use cache
let device_bases = params.get_or_upload_gpu_bases();
```

### 4. Batch Commit Pattern

Multiple polynomial commitments are batched to enable GPU pipelining:

```rust
// Collect polynomials, batch commit together
let all_polys: Vec<&Polynomial<F, LagrangeCoeff>> = collect_polys();
let commitments = batch_commit_refs::<F, CS>(params, &all_polys);
```

## Module Breakdown

### Core Modules

| File | Purpose | Key Changes |
|------|---------|-------------|
| `lib.rs` | Crate root | Added `gpu_accel` module, re-exports `init_gpu_backend`, conditional `unsafe` allow |
| `gpu_accel.rs` | **NEW** - GPU bridge | Re-exports all GPU types, provides wrapper functions |
| `utils/fft.rs` | **NEW** - GPU-aware FFT | `best_fft()` and `ifft()` with automatic GPU dispatch |
| `utils/mod.rs` | Utils module | Added `fft` submodule |

### KZG/Polynomial Commitment

| File | Purpose | Key Changes |
|------|---------|-------------|
| `poly/kzg/mod.rs` | KZG implementation | `commit()` and `commit_lagrange()` use cached GPU bases |
| `poly/kzg/msm.rs` | MSM operations | Added GPU MSM paths, async API, batch operations |
| `poly/kzg/params.rs` | KZG parameters | Added GPU base caches, lazy upload functions |
| `poly/batch_commit.rs` | **NEW** - Batch commits | GPU-pipelined batch commitments |
| `poly/mod.rs` | Poly module | Added `batch_commit` submodule |
| `poly/domain.rs` | FFT domain | Uses GPU-aware `best_fft` |

### PLONK Prover

| File | Purpose | Key Changes |
|------|---------|-------------|
| `plonk/prover.rs` | Main prover | Batch commits for advice, lookups, trash, permutations |
| `plonk/lookup/prover.rs` | Lookup argument | Separated `compute_product()` and `finalize()` |
| `plonk/permutation/prover.rs` | Permutation argument | Batch commits for Z polynomials |
| `plonk/trash/prover.rs` | Trash argument | Separated `compute()` and `finalize()` |

## Data Flow

### MSM Execution Path (Optimized)

```
1. ParamsKZG created with SRS
   └─▶ g_gpu, g_lagrange_gpu = OnceCell::new()  (empty caches)

2. First commit_lagrange() called
   ├─▶ should_use_gpu(size) checks threshold
   ├─▶ get_or_upload_gpu_lagrange_bases() triggered
   │   ├─▶ Convert G1Projective → G1Affine (batch_normalize)
   │   ├─▶ Reinterpret as ICICLE points (zero-copy, Montgomery form)
   │   └─▶ Upload to GPU (DeviceVec::copy_from_host_async)
   └─▶ msm_with_cached_bases() executes MSM on GPU

3. Subsequent commit_lagrange() calls
   ├─▶ get_or_upload_gpu_lagrange_bases() returns cached handle
   └─▶ msm_with_cached_bases() - no upload, direct computation
```

### Batch Commit Flow

```
1. Prover collects polynomials (advice, lookups, permutations)
   └─▶ polys: Vec<Polynomial<F, LagrangeCoeff>>

2. batch_commit_if_kzg_bls12() checks types at runtime
   ├─▶ size_of/align_of checks for ParamsKZG<Bls12>
   └─▶ is_fq::<F>() checks field type

3. If types match, transmute and call GPU batch:
   ├─▶ kzg_params.commit_lagrange_batch(&poly_refs)
   │   ├─▶ msm_batch_async() launches all MSMs
   │   ├─▶ GPU processes in pipeline (H2D overlap)
   │   └─▶ Wait for all results
   └─▶ Transmute results back to CS::Commitment

4. If types don't match, sequential CPU path
```

## Feature Flags

```toml
[features]
# Core GPU acceleration
gpu = ["midnight-bls12-381-cuda/gpu", "icicle-runtime", "once_cell"]
gpu-cuda = ["gpu"]  # Future: separate CUDA backend

# Tracing/instrumentation (dev only)
trace-msm = []      # Trace MSM operations
trace-fft = []      # Trace FFT operations
trace-kzg = []      # Trace KZG commits
trace-phases = []   # Trace prover phases
trace-all = ["trace-msm", "trace-fft", "trace-phases", "trace-kzg"]
```

## Performance Characteristics

### GPU Threshold

- **MSM**: GPU used when size ≥ 16,384 (K ≥ 14)
- **NTT**: GPU used when size ≥ 4,096 (K ≥ 12)
- Thresholds account for GPU transfer overhead

### Memory Layout Optimization

- Bases stored in **Montgomery form** on GPU
- Eliminates D2D copy + Montgomery conversion at MSM time
- ICICLE's `are_bases_montgomery_form = true` flag used

## Safety Considerations

### Unsafe Code Usage

The integration requires `unsafe` for:

1. **Type punning**: Reinterpreting `&[F]` as `&[Fq]` after type verification
2. **FFI boundaries**: Calling ICICLE CUDA kernels
3. **Memory transmutation**: Converting results back to generic types

All unsafe operations are:
- Guarded by runtime type checks (`TypeId`, `size_of`, `align_of`)
- Verified to maintain memory layout compatibility
- Documented with safety invariants

### Thread Safety

- `Arc<OnceCell<PrecomputedBases>>` ensures thread-safe lazy initialization
- GPU bases survive `Clone` operations (shared reference)
- ICICLE handles thread safety internally

## Testing

### Integration Tests

```bash
# Run all GPU tests
cargo test --features gpu --release -- --ignored

# Run specific benchmarks
cargo test --test gpu_msm_benchmark --features gpu --release -- --ignored --nocapture
```

### Test Files

| Test File | Purpose |
|-----------|---------|
| `gpu_integration.rs` | Basic GPU MSM correctness |
| `gpu_msm_benchmark.rs` | Performance comparisons |
| `e2e_proof_benchmark.rs` | Full proof generation |
| `msm_blst_detection.rs` | BLST detection tests |

## Migration Guide

### From CPU-only to GPU-enabled

1. Add feature flag to `Cargo.toml`:
   ```toml
   midnight-proofs = { version = "...", features = ["gpu"] }
   ```

2. Initialize GPU at startup (optional but recommended):
   ```rust
   if let Some(duration) = midnight_proofs::init_gpu_backend() {
       println!("GPU ready in {:?}", duration);
   }
   ```

3. Use existing APIs - GPU acceleration is automatic for:
   - `CS::commit_lagrange()`
   - `CS::commit()`
   - Domain FFT operations

## Changelog Summary

### Commit 274ee671: Port latest changes for GPU integration
- Added `bls12-381-cuda-backend` to workspace
- Added GPU dependencies to proofs crate
- Initial GPU MSM integration in `msm.rs`
- Batch commit helper functions in prover
