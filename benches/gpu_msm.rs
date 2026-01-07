use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use midnight_bls12_381_cuda::GpuMsmContext;
use midnight_curves::{Fq as Scalar, G1Affine};
use ff::Field;
use group::prime::PrimeCurveAffine;
use rand::rngs::OsRng;
use std::hint::black_box;

fn bench_gpu_msm_sizes(c: &mut Criterion) {
    // Check if GPU is available
    let ctx = match GpuMsmContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("GPU not available, skipping benchmarks: {}", e);
            return;
        }
    };

    let mut group = c.benchmark_group("GPU MSM");
    
    // Benchmark different MSM sizes (2^k) - limited for laptop GPU
    for k in [10, 12, 14] {
        let size = 1 << k;
        
        // Generate test data
        let scalars: Vec<Scalar> = (0..size)
            .map(|_| Scalar::random(&mut OsRng))
            .collect();
        
        let points: Vec<G1Affine> = (0..size)
            .map(|_| G1Affine::generator())
            .collect();
        
        // Upload bases to GPU once
        let device_bases = ctx.upload_g1_bases(&points)
            .expect("Failed to upload bases");
        
        // Benchmark synchronous MSM
        group.bench_with_input(
            BenchmarkId::new("sync", format!("2^{}", k)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(ctx.msm_with_device_bases(&scalars, &device_bases).unwrap())
                });
            }
        );
        
        // Benchmark async MSM
        group.bench_with_input(
            BenchmarkId::new("async", format!("2^{}", k)),
            &size,
            |b, _| {
                b.iter(|| {
                    let handle = ctx.msm_async(&scalars, &device_bases).unwrap();
                    black_box(handle.wait().unwrap())
                });
            }
        );
    }
    
    group.finish();
}

fn bench_gpu_msm_sequential_vs_potential_batch(c: &mut Criterion) {
    let ctx = match GpuMsmContext::new() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("Sequential MSM (baseline for batch)");
    
    let msm_size = 1 << 14; // 16K (reduced for laptop GPU memory)
    let batch_sizes = [4, 8];
    
    // Generate shared bases
    let points: Vec<G1Affine> = (0..msm_size)
        .map(|_| G1Affine::generator())
        .collect();
    let device_bases = ctx.upload_g1_bases(&points).unwrap();
    
    for &batch_size in &batch_sizes {
        // Generate batch of scalar sets
        let scalars_batch: Vec<Vec<Scalar>> = (0..batch_size)
            .map(|_| {
                (0..msm_size)
                    .map(|_| Scalar::random(&mut OsRng))
                    .collect()
            })
            .collect();
        
        // Benchmark sequential execution (current approach)
        group.bench_with_input(
            BenchmarkId::new("sequential", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    for scalars in &scalars_batch {
                        black_box(ctx.msm_with_device_bases(scalars, &device_bases).unwrap());
                    }
                });
            }
        );
    }
    
    group.finish();
}

fn bench_gpu_base_upload(c: &mut Criterion) {
    let ctx = match GpuMsmContext::new() {
        Ok(ctx) => ctx,
        Err(_) => return,
    };

    let mut group = c.benchmark_group("GPU Base Upload");
    
    for k in [12, 14] {
        let size = 1 << k;
        let points: Vec<G1Affine> = (0..size)
            .map(|_| G1Affine::generator())
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("upload", format!("2^{}", k)),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(ctx.upload_g1_bases(&points).unwrap())
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    name = benches;
    config = Criterion::default()
        .sample_size(10)  // Reduced for GPU benchmarks
        .measurement_time(std::time::Duration::from_secs(10));
    targets = bench_gpu_msm_sizes, bench_gpu_base_upload
);
criterion_main!(benches);
