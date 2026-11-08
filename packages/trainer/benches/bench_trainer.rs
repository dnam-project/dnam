#![forbid(unsafe_code)]
//! Criterion benchmark suite for `dnam-trainer` `uplo='U'` Gram accumulation and Cholesky solver.

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use dnam_trainer::solver::{
    accumulate_uplo_u_gram_and_cross, solve_uplo_u_cholesky_2ch, uplo_u_packed_len,
};

fn bench_uplo_u_cholesky_lite(c: &mut Criterion) {
    let n = 64usize;
    let mut g = vec![0.0_f64; uplo_u_packed_len(n)];
    let mut cross = vec![[0.0_f64; 2]; n];
    let mut row = vec![0.0_f64; n];
    for i in 0..n {
        row[i] = ((i as f64) * 0.13).sin();
    }
    accumulate_uplo_u_gram_and_cross(&mut g, &mut cross, &row, [0.5, -0.25], 1.0);

    c.bench_function("trainer_uplo_u_cholesky_64d", |b| {
        b.iter(|| {
            solve_uplo_u_cholesky_2ch(black_box(&g), black_box(&cross), black_box(1.0e-6))
                .expect("benchmark Cholesky solve must succeed")
        });
    });
}

criterion_group!(benches, bench_uplo_u_cholesky_lite);
criterion_main!(benches);
