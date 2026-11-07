//! Zero-copy `f64`/`f32` bindings for TORA Reference DSP Kernels (`Op-T`, `Op-M`, `Op-N`,
//! Control Bus, and Mixing, `ref[sdk.python.dsp.reference-kernels]`).

use dnam_engine::control::{
    eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64,
    eval_trust_region_softclip_f64,
};
use dnam_engine::math::algebraic_unit_rotor;
use dnam_engine::operators::reference::{
    catmull_rom_hermite_f64, cubic_lagrange_weights_f64, givens_householder_mix_8ch_f64,
    givens_householder_mix_16ch_f64, op_m_modal_step_f64, op_t_coercive_step_f64,
    op_t_tank_slice_f64, poly_adaa_algebraic_f64,
};
use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;

use crate::arrays::{slice_1d, slice_1d_mut, slice_2d_mut};
use crate::validation_err;

// ============================================================================
// 1. Op-T: Transport, Catmull-Rom & Coercive Tank Loop
// ============================================================================

/// 4-tap Catmull-Rom cubic Hermite fractional delay interpolation (`ref[dsp.op-t.reader.catmull-rom]`).
#[pyfunction]
pub fn native_catmull_rom_hermite(s_m1: f64, s_0: f64, s_1: f64, s_2: f64, mu: f64) -> f64 {
    catmull_rom_hermite_f64(s_m1, s_0, s_1, s_2, mu)
}

/// Vectorized 4-tap Catmull-Rom cubic Hermite evaluation over 1D `f64` slices.
#[pyfunction]
pub fn native_catmull_rom_hermite_slice<'py>(
    py: Python<'py>,
    s_m1: PyReadonlyArray1<'py, f64>,
    s_0: PyReadonlyArray1<'py, f64>,
    s_1: PyReadonlyArray1<'py, f64>,
    s_2: PyReadonlyArray1<'py, f64>,
    mu: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray1<'py, f64>,
) -> PyResult<()> {
    let n = s_0.len();
    if s_m1.len() != n || s_1.len() != n || s_2.len() != n || mu.len() != n || out.len() != n {
        return Err(validation_err(
            "catmull_rom_hermite_slice inputs and out must have identical length",
        ));
    }
    let sm1 = slice_1d(&s_m1, "s_m1")?;
    let s0 = slice_1d(&s_0, "s_0")?;
    let s1 = slice_1d(&s_1, "s_1")?;
    let s2 = slice_1d(&s_2, "s_2")?;
    let m = slice_1d(&mu, "mu")?;
    let o = slice_1d_mut(&mut out, "out")?;
    py.detach(|| {
        for i in 0..n {
            o[i] = catmull_rom_hermite_f64(sm1[i], s0[i], s1[i], s2[i], m[i]);
        }
    });
    Ok(())
}

/// Single-sample Op-T Coercive-Saturating Lyapunov loop step (`ref[dsp.stage4.loop-saturator]`).
#[pyfunction]
pub fn native_op_t_coercive_step(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    mut dc_x_prev: f64,
    mut dc_y_prev: f64,
) -> (f64, f64, f64) {
    let s_next = op_t_coercive_step_f64(
        u_in,
        g_lin,
        sigma_knee,
        v_ceil,
        r_dc,
        &mut dc_x_prev,
        &mut dc_y_prev,
    );
    (s_next, dc_x_prev, dc_y_prev)
}

/// Processes a contiguous `f64` slice through the reference Op-T coercive tank loop.
#[pyfunction]
pub fn native_op_t_tank_slice<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
    mut output: PyReadwriteArray1<'py, f64>,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> PyResult<()> {
    if input.len() != output.len() {
        return Err(validation_err(
            "input and output must have identical length",
        ));
    }
    let inp = slice_1d(&input, "input")?;
    let out = slice_1d_mut(&mut output, "output")?;
    py.detach(|| op_t_tank_slice_f64(inp, out, g_lin, sigma_knee, v_ceil, r_dc));
    Ok(())
}

// ============================================================================
// 2. Op-M: Symplectic Modal Step & Contractive Rotor
// ============================================================================

/// Single-sample Radial-Polar `Op-M` modal step returning `(y, next_z_re, next_z_im)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_op_m_modal_step(
    x: f64,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    mut z_re: f64,
    mut z_im: f64,
) -> (f64, f64, f64) {
    let y = op_m_modal_step_f64(
        x,
        r_low,
        dr_span,
        w_sel,
        b_gamma_norm,
        c_real,
        c_imag,
        u_rot_cos,
        u_rot_sin,
        &mut z_re,
        &mut z_im,
    );
    (y, z_re, z_im)
}

/// Processes a contiguous `f64` slice through the `Op-M` modal resonator, returning `(next_z_re, next_z_im)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_op_m_modal_slice<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
    mut output: PyReadwriteArray1<'py, f64>,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    mut z_re: f64,
    mut z_im: f64,
    reset: bool,
) -> PyResult<(f64, f64)> {
    if input.len() != output.len() {
        return Err(validation_err(
            "input and output must have identical length",
        ));
    }
    let inp = slice_1d(&input, "input")?;
    let out = slice_1d_mut(&mut output, "output")?;
    let (next_re, next_im) = py.detach(|| {
        if reset {
            z_re = 0.0;
            z_im = 0.0;
        }
        for (&x, y) in inp.iter().zip(out.iter_mut()) {
            *y = op_m_modal_step_f64(
                x,
                r_low,
                dr_span,
                w_sel,
                b_gamma_norm,
                c_real,
                c_imag,
                u_rot_cos,
                u_rot_sin,
                &mut z_re,
                &mut z_im,
            );
        }
        (z_re, z_im)
    });
    Ok((next_re, next_im))
}

/// Quadrant-reduced contractive unit-circle rotor returning `(cos_theta, sin_theta)`.
#[pyfunction]
pub fn native_unit_rotor(phi: f32) -> (f32, f32) {
    algebraic_unit_rotor(phi)
}

/// Vectorized unit-circle rotor evaluation over a 1D `f64` slice into `(2, N)`.
#[pyfunction]
pub fn native_unit_rotor_slice<'py>(
    py: Python<'py>,
    phi: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let p_slice = slice_1d(&phi, "phi")?;
    let out_slice = slice_2d_mut(&mut out, "out")?;
    let n = p_slice.len();
    if out_slice.len() != 2 * n {
        return Err(validation_err("unit_rotor output shape must be (2, N)"));
    }
    py.detach(|| {
        let (cos_row, sin_row) = out_slice.split_at_mut(n);
        for (i, &p_val) in p_slice.iter().enumerate() {
            let (c, s) = algebraic_unit_rotor(p_val as f32);
            cos_row[i] = f64::from(c);
            sin_row[i] = f64::from(s);
        }
    });
    Ok(())
}

// ============================================================================
// 3. Op-N: Division-Free Algebraic PolyADAA
// ============================================================================

/// Division-free algebraic `PolyADAA` segment average (`ref[dsp.op-n.polyadaa.midpoint]`).
#[pyfunction]
pub fn native_poly_adaa_algebraic(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta)
}

/// Single-sample `PolyADAA` step returning `(y, next_u_prev, next_a_prev, next_m_dc)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_poly_adaa_step(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
) -> (f64, f64, f64, f64) {
    let u_curr = ((x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
    let y_nl = 0.875 * a_curr + 0.125 * a_prev;
    let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
    let y = alpha * x + (y_nl - m_dc_curr);
    (y, u_curr, a_curr, m_dc_curr)
}

/// Processes a contiguous `f64` slice through `PolyADAA`, returning `(next_u_prev, next_a_prev, next_m_dc)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_poly_adaa_slice<'py>(
    py: Python<'py>,
    input: PyReadonlyArray1<'py, f64>,
    mut output: PyReadwriteArray1<'py, f64>,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    mut u_prev: f64,
    mut a_prev: f64,
    mut m_dc: f64,
    reset: bool,
) -> PyResult<(f64, f64, f64)> {
    if input.len() != output.len() {
        return Err(validation_err(
            "input and output must have identical length",
        ));
    }
    let inp = slice_1d(&input, "input")?;
    let out = slice_1d_mut(&mut output, "output")?;

    let (next_u, next_a, next_m) = py.detach(|| {
        if reset {
            let u_rest = (b_conv * (2.0 / 3.0)).clamp(-1.0, 1.0);
            let u2 = u_rest * u_rest;
            let g_rest = beta * u_rest * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2);
            u_prev = u_rest;
            a_prev = g_rest;
            m_dc = g_rest;
        }
        for (&x, y) in inp.iter().zip(out.iter_mut()) {
            let u_curr = ((x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
            let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
            let y_nl = 0.875 * a_curr + 0.125 * a_prev;
            let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
            u_prev = u_curr;
            a_prev = a_curr;
            m_dc = m_dc_curr;
            *y = alpha * x + (y_nl - m_dc_curr);
        }
        (u_prev, a_prev, m_dc)
    });
    Ok((next_u, next_a, next_m))
}

// ============================================================================
// 4. Control Bus Reference Kernels
// ============================================================================

/// Node-Anchored Newton/Lobatto basis evaluation (`ref[dsp.ctrl.basis.lobatto]`).
#[pyfunction]
pub fn native_eval_lobatto_basis(u: f64, anchors: [f64; 3], degree: usize) -> [f64; 3] {
    eval_newton_lobatto_basis_f64(u, anchors, degree)
}

/// Vectorized Node-Anchored Newton/Lobatto basis evaluation over a 1D `f64` slice into `(N, 3)`.
#[pyfunction]
pub fn native_eval_lobatto_basis_slice<'py>(
    py: Python<'py>,
    u: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
    anchors: [f64; 3],
    degree: usize,
) -> PyResult<()> {
    let u_slice = slice_1d(&u, "u")?;
    let out_slice = slice_2d_mut(&mut out, "out")?;
    let (out_rows, _) = out_slice.as_chunks_mut::<3>();
    if out_rows.len() != u_slice.len() {
        return Err(validation_err("LobattoBasis output shape must be (N, 3)"));
    }
    py.detach(|| {
        for (&u_val, row) in u_slice.iter().zip(out_rows.iter_mut()) {
            *row = eval_newton_lobatto_basis_f64(u_val, anchors, degree);
        }
    });
    Ok(())
}

/// Rank-4 Factorization Machine interaction evaluation (`ref[dsp.ctrl.accum.anova-fm]`).
#[pyfunction]
pub fn native_eval_rank4_fm<'py>(
    py: Python<'py>,
    features: PyReadonlyArray1<'py, f64>,
    v_factors_flat: PyReadonlyArray1<'py, f64>,
) -> PyResult<[f64; 4]> {
    let d = features.len();
    if v_factors_flat.len() != d * 4 {
        return Err(validation_err(format!(
            "v_factors must have shape ({d}, 4) = {} elements, got {}",
            d * 4,
            v_factors_flat.len()
        )));
    }
    let feat = slice_1d(&features, "features")?;
    let vf_slice = slice_1d(&v_factors_flat, "v_factors")?;
    let (v_rows, _) = vf_slice.as_chunks::<4>();
    Ok(py.detach(|| eval_rank4_fm_interaction_f64(feat, v_rows)))
}

/// 4-Harmonic Chebyshev LFO polynomial evaluation (`ref[dsp.op-n.ctrl.chebyshev-lfo]`).
#[pyfunction]
pub fn native_eval_chebyshev_lfo(cos_phi: f64) -> [f64; 4] {
    eval_chebyshev_lfo_4harm_f64(cos_phi)
}

/// Vectorized 4-Harmonic Chebyshev LFO evaluation over a 1D `f64` slice into `(N, 4)`.
#[pyfunction]
pub fn native_eval_chebyshev_lfo_slice<'py>(
    py: Python<'py>,
    cos_phi: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let c_slice = slice_1d(&cos_phi, "cos_phi")?;
    let out_slice = slice_2d_mut(&mut out, "out")?;
    let (out_rows, _) = out_slice.as_chunks_mut::<4>();
    if out_rows.len() != c_slice.len() {
        return Err(validation_err("chebyshev_lfo output shape must be (N, 4)"));
    }
    py.detach(|| {
        for (&c_val, row) in c_slice.iter().zip(out_rows.iter_mut()) {
            *row = eval_chebyshev_lfo_4harm_f64(c_val);
        }
    });
    Ok(())
}

/// Cubic trust-region soft-clipper (`ref[dsp.ctrl.op-n.trust-clip]`).
#[pyfunction]
pub fn native_eval_trust_region_softclip(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_f64(z, mh)
}

/// Vectorized cubic trust-region soft-clipper over a 1D `f64` slice.
#[pyfunction]
pub fn native_eval_trust_region_softclip_slice<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray1<'py, f64>,
    mh: f64,
) -> PyResult<()> {
    let z_slice = slice_1d(&z, "z")?;
    let out_slice = slice_1d_mut(&mut out, "out")?;
    if z_slice.len() != out_slice.len() {
        return Err(validation_err(
            "trust_region_softclip input and output lengths must match",
        ));
    }
    py.detach(|| {
        for (&z_val, y_out) in z_slice.iter().zip(out_slice.iter_mut()) {
            *y_out = eval_trust_region_softclip_f64(z_val, mh);
        }
    });
    Ok(())
}

// ============================================================================
// 5. Mixing & Interpolation Kernels
// ============================================================================

/// 4-tap Cubic Lagrange fractional delay weights (`ref[dsp.op-t.reader.catmull-rom]`).
#[pyfunction]
pub fn native_cubic_lagrange_weights(mu: f64) -> [f64; 4] {
    cubic_lagrange_weights_f64(mu)
}

/// Vectorized 4-tap Cubic Lagrange weights over a 1D `f64` slice into `(N, 4)`.
#[pyfunction]
pub fn native_cubic_lagrange_weights_slice<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let mu_slice = slice_1d(&mu, "mu")?;
    let out_slice = slice_2d_mut(&mut out, "out")?;
    let (out_rows, _) = out_slice.as_chunks_mut::<4>();
    if out_rows.len() != mu_slice.len() {
        return Err(validation_err(
            "cubic_lagrange_weights output shape must be (N, 4)",
        ));
    }
    py.detach(|| {
        for (&m, row) in mu_slice.iter().zip(out_rows.iter_mut()) {
            *row = cubic_lagrange_weights_f64(m);
        }
    });
    Ok(())
}

/// 8-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.stage4.spatial-tank]`).
#[pyfunction]
pub fn native_givens_householder_8ch(x: [f64; 8], cos_psi: f64, sin_psi: f64) -> [f64; 8] {
    givens_householder_mix_8ch_f64(&x, cos_psi, sin_psi)
}

/// 16-channel Orthogonal Givens + Householder mixing matrix (`ref[dsp.stage4.spatial-tank]`).
#[pyfunction]
pub fn native_givens_householder_16ch(
    x: [f64; 16],
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: [f64; 8],
    cross_sin: [f64; 8],
) -> [f64; 16] {
    givens_householder_mix_16ch_f64(&x, cos_psi, sin_psi, &cross_cos, &cross_sin)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_catmull_rom_hermite, m)?)?;
    m.add_function(wrap_pyfunction!(native_catmull_rom_hermite_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_op_t_coercive_step, m)?)?;
    m.add_function(wrap_pyfunction!(native_op_t_tank_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_op_m_modal_step, m)?)?;
    m.add_function(wrap_pyfunction!(native_op_m_modal_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_unit_rotor, m)?)?;
    m.add_function(wrap_pyfunction!(native_unit_rotor_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_poly_adaa_algebraic, m)?)?;
    m.add_function(wrap_pyfunction!(native_poly_adaa_step, m)?)?;
    m.add_function(wrap_pyfunction!(native_poly_adaa_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_lobatto_basis, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_lobatto_basis_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_rank4_fm, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_chebyshev_lfo, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_chebyshev_lfo_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_eval_trust_region_softclip, m)?)?;
    m.add_function(wrap_pyfunction!(
        native_eval_trust_region_softclip_slice,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(native_cubic_lagrange_weights, m)?)?;
    m.add_function(wrap_pyfunction!(native_cubic_lagrange_weights_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_givens_householder_8ch, m)?)?;
    m.add_function(wrap_pyfunction!(native_givens_householder_16ch, m)?)?;
    Ok(())
}
