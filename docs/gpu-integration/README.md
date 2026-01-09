# GPU Integration Documentation

This directory contains comprehensive documentation for the GPU acceleration integration in midnight-proofs.

## Documents

### [GPU_INTEGRATION_ARCHITECTURE.md](GPU_INTEGRATION_ARCHITECTURE.md)

High-level architecture overview including:
- Architecture diagram
- Design principles (single import point, type-safe dispatch, lazy caching)
- Module breakdown
- Data flow diagrams
- Feature flags
- Performance characteristics
- Safety considerations
- Testing guide
- Migration guide
- Changelog summary

### [FILE_CHANGES_DETAILED.md](FILE_CHANGES_DETAILED.md)

Detailed per-file documentation covering:
- Every file modified or added
- Code snippets showing before/after changes
- Explanation of each modification
- Function signatures and purposes
- Safety documentation for unsafe code

### [diffs/](diffs/)

Git diff files for every changed file, comparing against commit `2df59557c9ac3b79c355b29614158180064df48f` (midnight-proofs-v0.7.0):
- 21 diff files total
- ~3,130 lines added
- ~144 lines removed
- Includes new files, modified core files, and configuration changes

## Quick Reference

### Key New Files

| File | Purpose |
|------|---------|
| `gpu_accel.rs` | GPU bridge module - single import point |
| `batch_commit.rs` | GPU-pipelined batch commitments |
| `utils/fft.rs` | GPU-aware FFT operations |

### Key Modified Files

| File | Purpose |
|------|---------|
| `kzg/msm.rs` | GPU MSM with cached bases |
| `kzg/params.rs` | Lazy GPU base caching |
| `plonk/prover.rs` | Batch commit orchestration |

### Feature Flags

```toml
# Enable GPU acceleration
midnight-proofs = { features = ["gpu"] }

# Enable tracing (dev only)
midnight-proofs = { features = ["gpu", "trace-all"] }
```

### Quick Start

```rust
use midnight_proofs::init_gpu_backend;

fn main() {
    // Initialize GPU at startup (optional, reduces first-request latency)
    if let Some(duration) = init_gpu_backend() {
        println!("GPU ready in {:?}", duration);
    }
    
    // Use existing APIs - GPU acceleration is automatic
}
```

## Commits Covered

All changes merged from base commit `2df59557` (midnight-proofs-v0.7.0).
