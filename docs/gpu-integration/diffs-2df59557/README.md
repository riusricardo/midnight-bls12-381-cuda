# Git Diff Files Index

This directory contains git diff files for all changes made between commit `2df59557c9ac3b79c355b29614158180064df48f` (Release midnight-proofs-v0.7.0) and HEAD (commit `30a1a4d2` - Refac and modularize).

## How to Apply Diffs

To see what changed in a specific file:
```bash
cat <filename>.diff
```

To apply a diff (if reverting to original):
```bash
git apply --reverse <filename>.diff
```

## Diff Files

### New Files (Created)

| Diff File | Source Path | Lines Added |
|-----------|-------------|-------------|
| [gpu_accel.rs.diff](gpu_accel.rs.diff) | `proofs/src/gpu_accel.rs` | +372 |
| [batch_commit.rs.diff](batch_commit.rs.diff) | `proofs/src/poly/batch_commit.rs` | +223 |
| [utils_fft.rs.diff](utils_fft.rs.diff) | `proofs/src/utils/fft.rs` | +213 |
| [test_gpu_integration.rs.diff](test_gpu_integration.rs.diff) | `proofs/tests/gpu_integration.rs` | +182 |
| [test_gpu_msm_benchmark.rs.diff](test_gpu_msm_benchmark.rs.diff) | `proofs/tests/gpu_msm_benchmark.rs` | +270 |
| [test_e2e_proof_benchmark.rs.diff](test_e2e_proof_benchmark.rs.diff) | `proofs/tests/e2e_proof_benchmark.rs` | +262 |
| [test_msm_blst_detection.rs.diff](test_msm_blst_detection.rs.diff) | `proofs/tests/msm_blst_detection.rs` | +137 |

### Modified Files (Core)

| Diff File | Source Path | Changes |
|-----------|-------------|---------|
| [lib.rs.diff](lib.rs.diff) | `proofs/src/lib.rs` | GPU module exports, unsafe_code cfg |
| [kzg_msm.rs.diff](kzg_msm.rs.diff) | `proofs/src/poly/kzg/msm.rs` | GPU MSM operations |
| [kzg_params.rs.diff](kzg_params.rs.diff) | `proofs/src/poly/kzg/params.rs` | GPU base caching |
| [kzg_mod.rs.diff](kzg_mod.rs.diff) | `proofs/src/poly/kzg/mod.rs` | GPU commit paths |
| [plonk_prover.rs.diff](plonk_prover.rs.diff) | `proofs/src/plonk/prover.rs` | Batch commit orchestration |
| [lookup_prover.rs.diff](lookup_prover.rs.diff) | `proofs/src/plonk/lookup/prover.rs` | Refactored for batch commits |
| [trash_prover.rs.diff](trash_prover.rs.diff) | `proofs/src/plonk/trash/prover.rs` | Refactored for batch commits |
| [permutation_prover.rs.diff](permutation_prover.rs.diff) | `proofs/src/plonk/permutation/prover.rs` | Batch Z polynomial commits |
| [poly_domain.rs.diff](poly_domain.rs.diff) | `proofs/src/poly/domain.rs` | GPU-aware FFT import |

### Modified Files (Configuration)

| Diff File | Source Path | Changes |
|-----------|-------------|---------|
| [proofs_Cargo.toml.diff](proofs_Cargo.toml.diff) | `proofs/Cargo.toml` | GPU dependencies and features |
| [poly_mod.rs.diff](poly_mod.rs.diff) | `proofs/src/poly/mod.rs` | batch_commit module |
| [utils_mod.rs.diff](utils_mod.rs.diff) | `proofs/src/utils/mod.rs` | fft module |
| [gitignore.diff](gitignore.diff) | `.gitignore` | CUDA artifacts |
| [workspace_Cargo.toml.diff](workspace_Cargo.toml.diff) | `Cargo.toml` | bls12-381-cuda-backend member |

## Statistics

```
Total files changed: 21
Lines added: ~3,130
Lines removed: ~144
Net change: +2,986 lines
```

## Base Commit Information

- **Base Commit**: `2df59557c9ac3b79c355b29614158180064df48f`
- **Tag**: `midnight-proofs-v0.7.0`
- **Message**: "Release new version of several crates (#169)"

## Commits Included

1. `274ee671` - Port latest changes for GPU integration
2. `2992d093` - Integrate NTT/FFT GPU
3. `30a1a4d2` - Refac and modularize
