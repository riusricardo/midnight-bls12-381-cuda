# Detailed File Changes - GPU Integration

This document provides a detailed breakdown of every file modified or added in the GPU integration, explaining the purpose and implementation of each change.

---

## Table of Contents

1. [Root Configuration Files](#root-configuration-files)
2. [proofs/src/lib.rs](#proofssrclibrs)
3. [proofs/src/gpu_accel.rs (NEW)](#proofssrcgpu_accelrs-new)
4. [proofs/src/utils/fft.rs (NEW)](#proofssrcutilsfftrs-new)
5. [proofs/src/utils/mod.rs](#proofssrcutilsmodrs)
6. [proofs/src/poly/mod.rs](#proofssrcpolymodrs)
7. [proofs/src/poly/domain.rs](#proofssrcpolydomainrs)
8. [proofs/src/poly/batch_commit.rs (NEW)](#proofssrcpolybatch_commitrs-new)
9. [proofs/src/poly/kzg/mod.rs](#proofssrcpolykzgmodrs)
10. [proofs/src/poly/kzg/msm.rs](#proofssrcpolykzgmsmrs)
11. [proofs/src/poly/kzg/params.rs](#proofssrcpolykzgparamsrs)
12. [proofs/src/plonk/prover.rs](#proofssrcplonkproverrs)
13. [proofs/src/plonk/lookup/prover.rs](#proofssrcplonklookupproverrs)
14. [proofs/src/plonk/trash/prover.rs](#proofssrcplonktrashproverrs)
15. [proofs/src/plonk/permutation/prover.rs](#proofssrcplonkpermutationproverrs)
16. [Test Files](#test-files)

---

## Root Configuration Files

### Cargo.toml (Workspace Root)

**Change**: Added `bls12-381-cuda-backend` to workspace members.

```toml
members = ["proofs", "curves", "circuits", "aggregator", "zkir", "zk_stdlib", "bls12-381-cuda-backend"]
```

**Purpose**: Includes the GPU backend crate in the workspace for unified building and dependency management.

### .gitignore

**Changes**: Added ignores for:
- External repositories: `blst_repo/`, `icicle_repo/`
- GPU/CUDA artifacts: `*.so`, `*.nsys-rep`, `profile_data/`
- CUDA compilation files: `*.cubin`, `*.ptx`, `*.fatbin`
- Build directories: `bls12-381-cuda-backend/bls12-381/build/`

**Purpose**: Prevents CUDA compilation artifacts and large external repositories from being committed.

---

## proofs/src/lib.rs

**Location**: Root of the proofs crate

### Changes

1. **Conditional unsafe code allowance**:
   ```rust
   #![cfg_attr(not(feature = "gpu"), deny(unsafe_code))]
   #![cfg_attr(feature = "gpu", allow(unsafe_code))]
   ```
   **Purpose**: GPU integration requires unsafe for FFI and type transmutation. This allows unsafe only when GPU feature is enabled.

2. **New gpu_accel module**:
   ```rust
   #[cfg(feature = "gpu")]
   pub mod gpu_accel;
   ```
   **Purpose**: Declares the GPU bridge module.

3. **Re-exported init functions**:
   ```rust
   #[cfg(feature = "gpu")]
   pub use gpu_accel::init_gpu_backend;
   
   #[cfg(feature = "gpu")]
   pub use poly::kzg::msm::init_gpu_backend as init_gpu_backend_legacy;
   
   #[cfg(not(feature = "gpu"))]
   pub fn init_gpu_backend() -> Option<std::time::Duration> { None }
   ```
   **Purpose**: Provides easy access to GPU initialization from crate root with fallback stub for non-GPU builds.

---

## proofs/src/gpu_accel.rs (NEW)

**Location**: New file at `proofs/src/gpu_accel.rs`  
**Lines**: 372

### Purpose

This is the **GPU acceleration bridge** - the single import point for all GPU functionality. Other modules should NEVER import directly from `midnight_bls12_381_cuda` or `icicle_*`.

### Key Sections

#### 1. Re-exports from midnight-bls12-381-cuda

```rust
#[cfg(feature = "gpu")]
pub use midnight_bls12_381_cuda::{
    GpuMsmContext, PrecomputedBases, TypeConverter,
    msm::MsmHandle, BatchMsmHandle,
    ensure_backend_loaded,
    should_use_gpu as backend_should_use_gpu, should_use_gpu_batch,
    global_accelerator, GpuAccelerator, GpuCachedBases, MsmBackend, NttBackend,
    is_fq, is_g1_affine, is_g1_projective,
    should_dispatch_to_gpu_field, should_dispatch_to_gpu_curve, should_dispatch_to_gpu_ntt,
    try_as_fq_slice, try_as_fq_slice_mut, try_as_g1_affine_slice, try_as_g1_projective_slice,
    projective_to_curve,
    dispatch_msm, dispatch_ntt_inplace, dispatch_batch_msm, DispatchResult,
    with_fq_slice, with_fq_slice_mut, with_g1_affine_slice,
};
```

**Purpose**: Centralizes all GPU type imports. Other modules only need to `use crate::gpu_accel::*`.

#### 2. ICICLE Runtime Re-exports

```rust
#[cfg(feature = "gpu")]
pub use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
    Device as IcicleDevice,
    set_device as icicle_set_device,
};
```

**Purpose**: Needed by `params.rs` for direct GPU memory operations.

#### 3. init_gpu_backend()

```rust
pub fn init_gpu_backend() -> Option<std::time::Duration>
```

**Purpose**: Eagerly initializes GPU backend at application startup to avoid first-request latency. Returns warmup duration if successful.

#### 4. Availability Functions

```rust
pub fn is_gpu_available() -> bool
pub fn should_use_gpu(size: usize) -> bool
pub fn should_use_gpu_ntt(size: usize) -> bool
```

**Purpose**: Query functions for GPU availability and size thresholds.

#### 5. GpuBasesHandle

```rust
pub struct GpuBasesHandle {
    inner: Box<dyn GpuCachedBases>,
}
```

**Purpose**: Type-erased handle to GPU-cached bases for storage in proof parameters.

#### 6. MSM Operations

```rust
pub fn msm_auto(scalars: &[Fq], bases: &[G1Affine]) -> G1Projective
fn msm_blst(scalars: &[Fq], bases: &[G1Affine]) -> G1Projective
pub fn upload_bases(bases: &[G1Affine]) -> Result<GpuBasesHandle, String>
```

**Purpose**: High-level MSM functions with automatic GPU/CPU selection.

#### 7. NTT Operations

```rust
pub fn forward_ntt_auto(data: &mut [Fq]) -> Result<(), String>
pub fn inverse_ntt_auto(data: &mut [Fq]) -> Result<(), String>
fn forward_ntt_cpu(data: &mut [Fq]) -> Result<(), String>
fn inverse_ntt_cpu(data: &mut [Fq]) -> Result<(), String>
```

**Purpose**: NTT operations with GPU acceleration and CPU fallback.

---

## proofs/src/utils/fft.rs (NEW)

**Location**: New file at `proofs/src/utils/fft.rs`  
**Lines**: 213

### Purpose

GPU-aware FFT operations that automatically dispatch to GPU or CPU based on field type and size.

### Key Functions

#### 1. best_fft()

```rust
pub fn best_fft<F: PrimeField + 'static>(a: &mut [F], omega: F, log_n: u32)
```

**Behavior**:
- Uses `dispatch_ntt_inplace()` helper for type-safe routing
- For `Fq`: GPU if above threshold, else CPU
- For other fields: CPU only

**Implementation**:
```rust
#[cfg(feature = "gpu")]
{
    let used_gpu = dispatch_ntt_inplace(
        a,
        |fq_slice| midnight_bls12_381_cuda::ntt::forward_ntt_inplace_auto(fq_slice)
            .map_err(|e| e.to_string()),
        |data| midnight_curves::fft::best_fft(data, omega, log_n),
    );
    
    if used_gpu { return; }
}
// CPU fallback
midnight_curves::fft::best_fft(a, omega, log_n);
```

#### 2. ifft()

```rust
pub fn ifft<F: PrimeField + 'static>(a: &mut [F], omega_inv: F, log_n: u32, divisor: F)
```

**Note**: ICICLE's inverse NTT includes the 1/n scaling, so the CPU path applies divisor separately.

#### 3. gpu_fft_available()

```rust
pub fn gpu_fft_available() -> bool
```

**Purpose**: Query function for FFT GPU availability.

### Tests

Includes roundtrip tests verifying FFT → IFFT produces original values.

---

## proofs/src/utils/mod.rs

**Change**: Added single line:
```rust
pub mod fft;
```

**Purpose**: Exposes the new `fft` submodule.

---

## proofs/src/poly/mod.rs

**Change**: Added:
```rust
/// Batch commitment operations with GPU pipelining support
pub mod batch_commit;
```

**Purpose**: Exposes the new batch commit module.

---

## proofs/src/poly/domain.rs

**Change**: Modified FFT import:
```rust
// Before
use midnight_curves::fft::best_fft;

// After
use crate::utils::fft::best_fft;
```

**Purpose**: Domain operations now use GPU-aware FFT. This affects `lagrange_to_coeff()` and `coeff_to_extended_lagrange()`.

---

## proofs/src/poly/batch_commit.rs (NEW)

**Location**: New file at `proofs/src/poly/batch_commit.rs`  
**Lines**: 223

### Purpose

GPU-accelerated batch polynomial commitment. When using KZG with BLS12-381, multiple commits are launched asynchronously to overlap GPU computation with memory transfers.

### Key Functions

#### 1. batch_commit()

```rust
pub fn batch_commit<F, CS>(
    params: &CS::Parameters,
    polys: &[Polynomial<F, LagrangeCoeff>],
) -> Vec<CS::Commitment>
```

**Behavior**:
1. Single polynomial → direct commit
2. Multiple polynomials + KZG Bls12 + beneficial size → GPU batch
3. Otherwise → sequential CPU commits

**Type Detection**:
```rust
let params_match = size_of::<CS::Parameters>() == size_of::<ParamsKZG<Bls12>>()
    && align_of::<CS::Parameters>() == align_of::<ParamsKZG<Bls12>>();
let field_match = is_fq::<F>();
```

#### 2. batch_commit_refs()

Same as `batch_commit()` but accepts `&[&Polynomial<...>]` - more efficient when polynomials are already stored elsewhere.

### Safety

Uses `unsafe` transmutation after runtime type verification:
```rust
unsafe {
    let kzg_params = &*(params as *const CS::Parameters as *const ParamsKZG<Bls12>);
    let poly_refs: Vec<&Polynomial<Fq, LagrangeCoeff>> = polys.iter()
        .map(|p| &*(p as *const Polynomial<F, LagrangeCoeff>
                     as *const Polynomial<Fq, LagrangeCoeff>))
        .collect();
    kzg_params.commit_lagrange_batch(&poly_refs)
}
```

---

## proofs/src/poly/kzg/mod.rs

**Location**: KZG polynomial commitment scheme implementation

### Changes

#### 1. commit() - Enhanced with GPU caching

**Before**:
```rust
fn commit(params, polynomial) {
    let mut scalars = Vec::with_capacity(polynomial.len());
    scalars.extend(polynomial.iter());
    msm_specific::<E::G1Affine>(&scalars, &params.g[..size])
}
```

**After**:
```rust
fn commit(params, polynomial) {
    let scalars: &[E::Fr] = &**polynomial;  // No allocation!
    
    #[cfg(feature = "gpu")]
    if should_use_gpu(size) {
        let device_bases = params.get_or_upload_gpu_bases();
        return msm_with_cached_bases::<E::G1Affine>(scalars, device_bases);
    }
    
    msm_specific::<E::G1Affine>(scalars, &params.g[..size])
}
```

**Improvements**:
- Removed Vec allocation (use Deref on Polynomial)
- GPU path uses cached bases (no per-call upload)
- Optional tracing for performance analysis

#### 2. commit_lagrange() - Same pattern as commit()

Uses `get_or_upload_gpu_lagrange_bases()` for Lagrange form bases.

#### 3. Async Commitment Helpers (GPU-only)

```rust
pub fn commit_lagrange_async<E: MultiMillerLoop>(
    params: &ParamsKZG<E>,
    poly: &Polynomial<E::Fr, LagrangeCoeff>,
) -> Result<MsmHandle, Error>

pub fn commit_lagrange_batch_async<E: MultiMillerLoop>(
    params: &ParamsKZG<E>,
    polys: &[&Polynomial<E::Fr, LagrangeCoeff>],
) -> Result<Vec<MsmHandle>, Error>
```

**Purpose**: Launch MSMs without waiting, enabling CPU/GPU overlap and pipelining.

---

## proofs/src/poly/kzg/msm.rs

**Location**: Multi-Scalar Multiplication implementations  
**Changes**: +500 lines

### Key Additions

#### 1. Global GPU MSM Context

```rust
static GLOBAL_MSM_CONTEXT: OnceLock<GpuMsmContext> = OnceLock::new();

pub fn get_msm_context() -> &'static GpuMsmContext
pub fn init_gpu_backend() -> Option<std::time::Duration>
```

**Purpose**: Lazy-initialized singleton for GPU MSM operations.

#### 2. is_blst_available()

```rust
pub fn is_blst_available<C: CurveAffine + 'static>() -> bool
```

**Purpose**: Check if BLST optimization is available for given curve type.

#### 3. msm_with_cached_bases()

```rust
pub fn msm_with_cached_bases<C: CurveAffine + 'static>(
    coeffs: &[C::Scalar],
    device_bases: &PrecomputedBases,
) -> C::Curve
```

**Purpose**: MSM using pre-uploaded GPU bases. This is the **primary optimization** - uses bases cached in GPU memory, eliminating per-call conversion and upload overhead.

#### 4. msm_specific() - Enhanced

**Before**: Simple BLST call or fallback

**After**:
```rust
pub fn msm_specific<C: CurveAffine>(coeffs, bases) -> C::Curve {
    // 1. Check if BLST available, else generic fallback
    if !is_blst_available::<C>() { 
        return generic_msm(coeffs, bases); 
    }
    
    // 2. GPU path for large MSMs
    #[cfg(feature = "gpu")]
    if should_use_gpu(size) {
        let ctx = get_msm_context();
        let bases_affine = convert_to_affine(bases);
        if let Ok(res) = ctx.msm(coeffs, &bases_affine) {
            return res;
        }
        // Fall through to BLST on error
    }
    
    // 3. BLST path for small MSMs or non-GPU
    G1Projective::multi_exp(bases, coeffs)
}
```

#### 5. Async MSM API

```rust
pub fn msm_with_cached_bases_async<C>(coeffs, device_bases) -> Result<MsmHandle, Error>
pub fn msm_batch_async<C>(coeffs_batch, device_bases) -> Result<Vec<MsmHandle>, Error>
```

**Purpose**: Launch async MSMs for CPU/GPU overlap.

#### 6. Batch MSM Operations

```rust
pub fn msm_batch_with_cached_bases<C>(coeffs_batch, device_bases) -> Vec<C::Curve>
pub fn msm_batch_with_cached_bases_async<C>(coeffs_batch, device_bases) -> Result<BatchMsmHandleWrapper<C>, Error>
pub fn msm_batch_pipelined<C>(coeffs_batch, device_bases) -> Vec<C::Curve>
```

**Purpose**: Multiple MSMs with shared bases in optimized patterns.

#### 7. BatchMsmHandleWrapper

```rust
pub struct BatchMsmHandleWrapper<C: CurveAffine> {
    handle: BatchMsmHandle,
    _phantom: PhantomData<C>,
}
```

**Purpose**: Type-safe wrapper for batch async handles.

---

## proofs/src/poly/kzg/params.rs

**Location**: KZG parameter storage  
**Changes**: +200 lines

### Key Changes

#### 1. GPU Base Caches Added to ParamsKZG

```rust
pub struct ParamsKZG<E: Engine> {
    pub(crate) g: Vec<E::G1>,
    pub(crate) g_lagrange: Vec<E::G1>,
    pub(crate) g2: E::G2,
    pub(crate) s_g2: E::G2,
    
    // NEW: GPU-cached bases
    #[cfg(feature = "gpu")]
    pub(crate) g_gpu: Arc<OnceCell<PrecomputedBases>>,
    #[cfg(feature = "gpu")]
    pub(crate) g_lagrange_gpu: Arc<OnceCell<PrecomputedBases>>,
}
```

**Purpose**: Lazy-initialized GPU base storage. Uses `Arc<OnceCell<>>` so:
- Thread-safe lazy initialization
- Survives Clone (shared reference to same GPU memory)

#### 2. Custom Debug Implementation

```rust
impl<E: Engine> Debug for ParamsKZG<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParamsKZG")
            .field("g_len", &self.g.len())
            .field("g_lagrange_len", &self.g_lagrange.len())
            .finish()
    }
}
```

**Purpose**: GPU caches contain non-Debug types, so custom impl needed.

#### 3. get_or_upload_gpu_bases()

```rust
#[cfg(feature = "gpu")]
pub fn get_or_upload_gpu_bases(&self) -> &PrecomputedBases {
    self.g_gpu.get_or_init(|| {
        // 1. Ensure ICICLE backend loaded
        ensure_backend_loaded().expect("Failed to load ICICLE backend");
        
        // 2. Set CUDA device
        let device = IcicleDevice::new("CUDA", 0);
        icicle_set_device(&device).expect("Failed to set GPU device");
        
        // 3. Convert Projective → Affine
        let mut bases_affine = vec![E::G1Affine::identity(); self.g.len()];
        E::G1::batch_normalize(&self.g, &mut bases_affine);
        
        // 4. Zero-copy view as ICICLE points (Montgomery form preserved!)
        let bases_midnight: &[midnight_curves::G1Affine] = unsafe {
            std::mem::transmute(bases_affine.as_slice())
        };
        let icicle_points = TypeConverter::g1_slice_as_icicle(bases_midnight);
        
        // 5. Upload to GPU
        let stream = IcicleStream::default();
        let mut device_bases = DeviceVec::device_malloc_async(icicle_points.len(), &stream)
            .expect("Failed to allocate GPU memory");
        device_bases.copy_from_host_async(HostSlice::from_slice(icicle_points), &stream)
            .expect("Failed to upload bases to GPU");
        stream.synchronize().expect("Sync failed");
        
        PrecomputedBases::new(device_bases, icicle_points.len())
    })
}
```

**Key Optimization**: Bases uploaded in **Montgomery form** (the internal representation of midnight-curves). This eliminates per-MSM D2D copy + Montgomery conversion.

#### 4. get_or_upload_gpu_lagrange_bases()

Same pattern as above, but for Lagrange form bases (`g_lagrange`).

#### 5. commit_lagrange_batch()

```rust
#[cfg(feature = "gpu")]
pub fn commit_lagrange_batch(
    &self,
    polys: &[&Polynomial<E::Fr, LagrangeCoeff>],
) -> Vec<E::G1>
```

**Purpose**: Production-ready batch commit with GPU pipelining. Handles:
- Empty input
- Single polynomial optimization
- Async vs sync path selection based on size and count
- GPU/CPU fallback

---

## proofs/src/plonk/prover.rs

**Location**: Main PLONK prover  
**Changes**: +140 lines

### Key Changes

#### 1. Batch Commit Helper Functions

```rust
#[cfg(feature = "gpu")]
fn batch_commit_refs_if_kzg_bls12<F, CS>(
    params: &CS::Parameters,
    polys: &[&Polynomial<F, LagrangeCoeff>],
) -> Option<Vec<CS::Commitment>>

#[cfg(feature = "gpu")]
fn batch_commit_if_kzg_bls12<F, CS>(
    params: &CS::Parameters,
    polys: &[Polynomial<F, LagrangeCoeff>],
) -> Option<Vec<CS::Commitment>>
```

**Purpose**: Try GPU batch optimization, return `None` to fall back to sequential.

**Type Detection**:
```rust
let params_match = size_of::<CS::Parameters>() == size_of::<ParamsKZG<Bls12>>()
    && align_of::<CS::Parameters>() == align_of::<ParamsKZG<Bls12>>();
let field_match = TypeId::of::<F>() == TypeId::of::<Fq>();
```

#### 2. Lookup Commitments - Restructured

**Before**: Compute product and commit in single `commit_product()` call

**After**:
```rust
// Step 1: Compute all products (no commits yet)
let lookup_products: Vec<Vec<ProductComputed<F>>> = lookups.into_iter()
    .map(|lookups| {
        lookups.into_iter()
            .map(|lookup| lookup.compute_product::<CS>(pk, beta, gamma, &mut rng))
            .collect()
    })
    .collect();

// Step 2: Batch commit all products
#[cfg(feature = "gpu")]
let lookups: Vec<Vec<Committed<F>>> = {
    let all_polys: Vec<&Polynomial<...>> = lookup_products.iter()
        .flat_map(|products| products.iter().map(|p| &p.product_poly_lagrange))
        .collect();
    
    if let Some(commits) = batch_commit_refs_if_kzg_bls12::<F, CS>(params, &all_polys) {
        // Distribute commits back and finalize
        ...
    } else {
        // Sequential fallback
        ...
    }
};
```

#### 3. Trash Commitments - Same Pattern

Separated `compute()` and `finalize()` for batch committing.

#### 4. Advice Commitments - Batch Optimized

```rust
let advice_commitments: Vec<_> = {
    #[cfg(feature = "gpu")]
    {
        if let Some(batch_commits) = batch_commit_if_kzg_bls12::<F, CS>(params, &advice_values) {
            batch_commits
        } else {
            advice_values.iter().map(|poly| CS::commit_lagrange(params, poly)).collect()
        }
    }
    #[cfg(not(feature = "gpu"))]
    {
        advice_values.iter().map(|poly| CS::commit_lagrange(params, poly)).collect()
    }
};
```

---

## proofs/src/plonk/lookup/prover.rs

**Location**: Lookup argument prover  
**Changes**: Restructured for batch commits

### Key Changes

#### 1. New ProductComputed Struct

```rust
#[derive(Debug)]
pub(crate) struct ProductComputed<F: PrimeField> {
    pub(crate) permuted_input_poly: Polynomial<F, Coeff>,
    pub(crate) permuted_table_poly: Polynomial<F, Coeff>,
    pub(crate) product_poly_lagrange: Polynomial<F, LagrangeCoeff>,
}
```

**Purpose**: Intermediate state holding computed polynomial before commitment.

#### 2. Refactored commit_product() → compute_product()

**Before**:
```rust
pub fn commit_product<CS, T>(self, pk, params, beta, gamma, rng, transcript) 
    -> Result<Committed<F>, Error>
{
    // ... compute product polynomial ...
    let product_commitment = CS::commit_lagrange(params, &z);
    let z = pk.vk.domain.lagrange_to_coeff(z);
    transcript.write(&product_commitment)?;
    Ok(Committed { ... })
}
```

**After**:
```rust
pub fn compute_product<CS>(self, pk, beta, gamma, rng) -> ProductComputed<F> {
    // ... compute product polynomial (same logic) ...
    ProductComputed {
        permuted_input_poly: self.permuted_input_poly,
        permuted_table_poly: self.permuted_table_poly,
        product_poly_lagrange: z,
    }
}
```

#### 3. New finalize() Method

```rust
impl<F: PrimeField> ProductComputed<F> {
    pub fn finalize<CS, T>(self, pk, commitment, transcript) -> Result<Committed<F>, Error> {
        let product_poly = pk.vk.domain.lagrange_to_coeff(self.product_poly_lagrange);
        transcript.write(&commitment)?;
        Ok(Committed {
            permuted_input_poly: self.permuted_input_poly,
            permuted_table_poly: self.permuted_table_poly,
            product_poly,
        })
    }
}
```

**Purpose**: Accepts pre-computed commitment (from batch commit) and finalizes the structure.

---

## proofs/src/plonk/trash/prover.rs

**Location**: Trash argument prover  
**Changes**: Same restructuring pattern as lookup

### Key Changes

#### 1. New TrashComputed Struct

```rust
#[derive(Debug)]
pub(crate) struct TrashComputed<F: PrimeField> {
    pub(crate) trash_poly_lagrange: Polynomial<F, LagrangeCoeff>,
}
```

#### 2. Refactored commit() → compute()

**Before**:
```rust
pub fn commit<CS, T>(self, params, domain, ..., transcript) -> Result<Committed<F>, Error> {
    let compressed_expression = /* compute */;
    let trash_commitment = CS::commit_lagrange(params, &compressed_expression);
    let trash_poly = domain.lagrange_to_coeff(compressed_expression);
    transcript.write(&trash_commitment)?;
    Ok(Committed { trash_poly })
}
```

**After**:
```rust
pub fn compute(&self, domain, ...) -> TrashComputed<F> {
    let compressed_expression = /* compute */;
    TrashComputed { trash_poly_lagrange: compressed_expression }
}
```

#### 3. New finalize() Method

```rust
impl<F: PrimeField> TrashComputed<F> {
    pub fn finalize<CS, T>(self, domain, commitment, transcript) -> Result<Committed<F>, Error> {
        let trash_poly = domain.lagrange_to_coeff(self.trash_poly_lagrange);
        transcript.write(&commitment)?;
        Ok(Committed { trash_poly })
    }
}
```

---

## proofs/src/plonk/permutation/prover.rs

**Location**: Permutation argument prover

### Key Changes

#### 1. Collect Z Polynomials Before Committing

**Before**: Commit each Z polynomial inside the loop

**After**:
```rust
let mut z_polys_lagrange: Vec<Polynomial<F, LagrangeCoeff>> = vec![];

for (columns, permutations) in chunks {
    // ... compute Z polynomial ...
    z_polys_lagrange.push(z);
}
```

#### 2. Batch Commit All Z Polynomials

```rust
#[cfg(feature = "gpu")]
let commitments = {
    let poly_refs: Vec<_> = z_polys_lagrange.iter().collect();
    batch_commit_refs::<F, CS>(params, &poly_refs)
};

#[cfg(not(feature = "gpu"))]
let commitments: Vec<_> = z_polys_lagrange.iter()
    .map(|z| CS::commit_lagrange(params, z))
    .collect();
```

#### 3. Write Commitments and Build Sets

```rust
for (z, commitment) in z_polys_lagrange.into_iter().zip(commitments.into_iter()) {
    let permutation_product_poly = domain.lagrange_to_coeff(z);
    transcript.write(&commitment)?;
    sets.push(CommittedSet { permutation_product_poly });
}
```

---

## Test Files

### proofs/tests/gpu_integration.rs (NEW)

**Lines**: 182

**Tests**:
- `test_gpu_msm_small` - 1024 points
- `test_gpu_msm_k14` - 16384 points
- `test_gpu_msm_k16` - 65536 points
- `test_gpu_context_creation`
- `test_gpu_msm_simple_identity` - Verifies n*G computation
- `test_gpu_msm_empty`
- G2 variants of above tests
- `test_gpu_warmup`

**Purpose**: Verify GPU MSM correctness against CPU reference.

### proofs/tests/gpu_msm_benchmark.rs (NEW)

**Lines**: 270

**Purpose**: Performance benchmark comparing GPU vs CPU MSM.

**Key Function**:
```rust
fn benchmark_msm(size: usize, ctx: &GpuMsmContext) -> (gpu_warmup_ms, gpu_cached_ms, cpu_ms, speedup)
```

**Tests**:
- `gpu_msm_benchmark` - Full benchmark with table output
- `gpu_msm_benchmark_quick` - CI-friendly quick benchmark

### proofs/tests/e2e_proof_benchmark.rs (NEW)

**Lines**: 262

**Purpose**: End-to-end proof generation benchmark with GPU.

### proofs/tests/msm_blst_detection.rs (NEW)

**Lines**: 137

**Purpose**: Tests for BLST detection and type checking.

---

## proofs/Cargo.toml

### Added Dependencies

```toml
# GPU acceleration
midnight-bls12-381-cuda = { path = "../bls12-381-cuda-backend", optional = true }
icicle-runtime = { git = "...", tag = "v4.0.0", optional = true }
once_cell = { version = "1.20", optional = true }
static_assertions = "1.1"
```

### Added Features

```toml
[features]
# Tracing
trace-msm = []
trace-fft = []
trace-phases = []
trace-kzg = []
trace-all = ["trace-msm", "trace-fft", "trace-phases", "trace-kzg"]

# GPU acceleration
gpu = ["midnight-bls12-381-cuda/gpu", "icicle-runtime", "once_cell"]
gpu-cuda = ["gpu"]
```
