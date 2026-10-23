//! Zero-copy `f64`/`f32` bindings to Core 1, Core 2, Control Bus, and Math reference DSP kernels.

use dnam_engine::control::{
    eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64,
    eval_trust_region_softclip_f64,
};
use dnam_engine::core1::{
    poly_adaa_algebraic_f64, poly_adaa_slice_f64, poly_adaa_step_f64, selective_rglru_slice_f64,
    selective_rglru_step_f64,
};
use dnam_engine::core2::{
    cubic_lagrange_weights_f64, givens_householder_mix_8ch_f64, mufdn_coercive_step_f64,
    mufdn_slice_f64,
};
use dnam_engine::math::algebraic_unit_rotor;
use numpy::{PyReadonlyArray1, PyReadwriteArray1, PyReadwriteArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Division-free algebraic `PolyADAA` segment average (`CL-01`).
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
    mut u_prev: f64,
    mut a_prev: f64,
    mut m_dc: f64,
) -> (f64, f64, f64, f64) {
    let u_curr = ((x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
    let y_nl = 0.875 * a_curr + 0.125 * a_prev;
    let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
    u_prev = u_curr;
    a_prev = a_curr;
    m_dc = m_dc_curr;
    let y = if b_conv == 0.0 {
        let mut up = u_prev;
        let mut ap = a_prev;
        let mut md = m_dc;
        let _ = poly_adaa_step_f64(x, alpha, beta, delta, r_ac, &mut up, &mut ap, &mut md);
        alpha * x + (y_nl - m_dc_curr)
    } else {
        alpha * x + (y_nl - m_dc_curr)
    };
    (y, u_prev, a_prev, m_dc)
}

/// Processes a contiguous `f64` slice through `PolyADAA`, starting from `(u_prev, a_prev, m_dc)`
/// (or analytical DC resting equilibrium if `reset == true`) and returning `(next_u_prev, next_a_prev, next_m_dc)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_poly_adaa_scan<'py>(
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
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] input and output must have identical length",
        ));
    }
    let inp = input
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] input must be contiguous"))?;
    let out = output
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] output must be contiguous"))?;

    if reset {
        poly_adaa_slice_f64(inp, out, b_conv, alpha, beta, delta, r_ac);
        let u_rest = (b_conv * (2.0 / 3.0)).clamp(-1.0, 1.0);
        let u2 = u_rest * u_rest;
        let g_rest = beta * u_rest * (1.5 - 0.5 * u2) + delta * (u2 - u2 * u2);
        u_prev = u_rest;
        a_prev = g_rest;
        m_dc = g_rest;
        for &x in inp {
            let u_curr = ((x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
            let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
            let y_nl = 0.875 * a_curr + 0.125 * a_prev;
            let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
            u_prev = u_curr;
            a_prev = a_curr;
            m_dc = m_dc_curr;
        }
    } else {
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
    }
    Ok((u_prev, a_prev, m_dc))
}

/// Single-sample Radial-Polar `SelectiveRgLru` step returning `(y, next_z_re, next_z_im)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_selective_rglru_step(
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
    let y = selective_rglru_step_f64(
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

/// Processes a contiguous `f64` slice through `SelectiveRgLru`, starting from `(z_re, z_im)`
/// and returning `(next_z_re, next_z_im)`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_selective_rglru_scan<'py>(
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
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] input and output must have identical length",
        ));
    }
    let inp = input
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] input must be contiguous"))?;
    let out = output
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] output must be contiguous"))?;

    if reset && z_re == 0.0 && z_im == 0.0 {
        selective_rglru_slice_f64(
            inp,
            out,
            r_low,
            dr_span,
            w_sel,
            b_gamma_norm,
            c_real,
            c_imag,
            u_rot_cos,
            u_rot_sin,
        );
        z_re = 0.0;
        z_im = 0.0;
        for &x in inp {
            let _ = selective_rglru_step_f64(
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
    } else {
        if reset {
            z_re = 0.0;
            z_im = 0.0;
        }
        for (&x, y) in inp.iter().zip(out.iter_mut()) {
            *y = selective_rglru_step_f64(
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
    }
    Ok((z_re, z_im))
}

/// 4-FMA + Algebraic Newton unit-circle rotor returning `(cos_theta, sin_theta)`.
#[pyfunction]
pub fn native_unit_rotor(phi: f32) -> (f32, f32) {
    algebraic_unit_rotor(phi)
}

/// 4-tap Cubic Lagrange fractional delay weights (`CL-06`).
#[pyfunction]
pub fn native_cubic_lagrange_weights(mu: f64) -> [f64; 4] {
    cubic_lagrange_weights_f64(mu)
}

/// 8-channel Orthogonal Givens + Householder mixing matrix `U(psi)` (`CL-08`).
#[pyfunction]
pub fn native_givens_householder_mix_8ch(x: [f64; 8], cos_psi: f64, sin_psi: f64) -> [f64; 8] {
    givens_householder_mix_8ch_f64(&x, cos_psi, sin_psi)
}

/// Single-sample Coercive-Saturating Lyapunov loop step (`CL-09`..`CL-11`).
#[pyfunction]
pub fn native_mufdn_coercive_step(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    mut dc_x_prev: f64,
    mut dc_y_prev: f64,
) -> (f64, f64, f64) {
    let s_next = mufdn_coercive_step_f64(
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

/// Processes a contiguous `f64` slice through a reference Coercive-Saturating `M-UFDN` loop.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn native_mufdn_slice<'py>(
    input: PyReadonlyArray1<'py, f64>,
    mut output: PyReadwriteArray1<'py, f64>,
    delay_samples: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> PyResult<()> {
    if input.len() != output.len() {
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] input and output must have identical length",
        ));
    }
    let inp = input
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] input must be contiguous"))?;
    let out = output
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] output must be contiguous"))?;
    mufdn_slice_f64(inp, out, delay_samples, g_lin, sigma_knee, v_ceil, r_dc);
    Ok(())
}

/// Node-Anchored Newton/Lobatto basis evaluation (`CL-13`).
#[pyfunction]
pub fn native_eval_lobatto_basis(u: f64, anchors: [f64; 3], degree: usize) -> [f64; 3] {
    eval_newton_lobatto_basis_f64(u, anchors, degree)
}

/// Vectorized Node-Anchored Newton/Lobatto basis evaluation over a 1D `f64` slice into `(N, 3)`.
#[pyfunction]
pub fn native_eval_lobatto_basis_slice<'py>(
    u: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
    anchors: [f64; 3],
    degree: usize,
) -> PyResult<()> {
    let u_slice = u
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] u must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] out must be contiguous"))?;
    let (out_rows, _) = out_slice.as_chunks_mut::<3>();
    if out_rows.len() != u_slice.len() {
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] LobattoBasis output shape must be (N, 3)",
        ));
    }
    for (&u_val, row) in u_slice.iter().zip(out_rows.iter_mut()) {
        *row = eval_newton_lobatto_basis_f64(u_val, anchors, degree);
    }
    Ok(())
}

/// Rank-4 Factorization Machine interaction evaluation (`CL-14`).
#[pyfunction]
pub fn native_eval_rank4_fm<'py>(
    features: PyReadonlyArray1<'py, f64>,
    v_factors_flat: PyReadonlyArray1<'py, f64>,
) -> PyResult<[f64; 4]> {
    let d = features.len();
    if v_factors_flat.len() != d * 4 {
        return Err(PyValueError::new_err(format!(
            "[DNAM-VALIDATION] v_factors must have shape ({d}, 4) = {} elements, got {}",
            d * 4,
            v_factors_flat.len()
        )));
    }
    let feat = features
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] features must be contiguous"))?;
    let vf_slice = v_factors_flat
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] v_factors must be contiguous"))?;

    let (v_rows, _) = vf_slice.as_chunks::<4>();
    Ok(eval_rank4_fm_interaction_f64(feat, v_rows))
}

/// 4-Harmonic Chebyshev LFO polynomial evaluation (`CL-16`).
#[pyfunction]
pub fn native_eval_chebyshev_lfo(cos_phi: f64) -> [f64; 4] {
    eval_chebyshev_lfo_4harm_f64(cos_phi)
}

/// Vectorized 4-Harmonic Chebyshev LFO evaluation over a 1D `f64` slice into `(N, 4)`.
#[pyfunction]
pub fn native_eval_chebyshev_lfo_slice<'py>(
    cos_phi: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let c_slice = cos_phi
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] cos_phi must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] out must be contiguous"))?;
    let (out_rows, _) = out_slice.as_chunks_mut::<4>();
    if out_rows.len() != c_slice.len() {
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] chebyshev_lfo output shape must be (N, 4)",
        ));
    }
    for (&c_val, row) in c_slice.iter().zip(out_rows.iter_mut()) {
        *row = eval_chebyshev_lfo_4harm_f64(c_val);
    }
    Ok(())
}

/// Cubic trust-region soft-clipper (`CL-15`).
#[pyfunction]
pub fn native_eval_trust_region_softclip(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_f64(z, mh)
}

/// Vectorized cubic trust-region soft-clipper over a 1D `f64` slice.
#[pyfunction]
pub fn native_eval_trust_region_softclip_slice<'py>(
    z: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray1<'py, f64>,
    mh: f64,
) -> PyResult<()> {
    let z_slice = z
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] z must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] out must be contiguous"))?;
    if z_slice.len() != out_slice.len() {
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] trust_region_softclip input and output lengths must match",
        ));
    }
    for (&z_val, y_out) in z_slice.iter().zip(out_slice.iter_mut()) {
        *y_out = eval_trust_region_softclip_f64(z_val, mh);
    }
    Ok(())
}

/// Vectorized $3\text{ kHz}$ 4-FMA + Newton unit-circle rotor evaluation over a 1D `f64` slice into `(2, N)`.
#[pyfunction]
pub fn native_unit_rotor_slice<'py>(
    phi: PyReadonlyArray1<'py, f64>,
    mut out: PyReadwriteArray2<'py, f64>,
) -> PyResult<()> {
    let p_slice = phi
        .as_slice()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] phi must be contiguous"))?;
    let out_slice = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("[DNAM-VALIDATION] out must be contiguous"))?;
    let n = p_slice.len();
    if out_slice.len() != 2 * n {
        return Err(PyValueError::new_err(
            "[DNAM-VALIDATION] unit_rotor output shape must be (2, N)",
        ));
    }
    let (cos_row, sin_row) = out_slice.split_at_mut(n);
    for (i, &p_val) in p_slice.iter().enumerate() {
        let (c, s) = algebraic_unit_rotor(p_val as f32);
        cos_row[i] = f64::from(c);
        sin_row[i] = f64::from(s);
    }
    Ok(())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(native_poly_adaa_algebraic, m)?)?;
    m.add_function(wrap_pyfunction!(native_poly_adaa_step, m)?)?;
    m.add_function(wrap_pyfunction!(native_poly_adaa_scan, m)?)?;
    m.add_function(wrap_pyfunction!(native_selective_rglru_step, m)?)?;
    m.add_function(wrap_pyfunction!(native_selective_rglru_scan, m)?)?;
    m.add_function(wrap_pyfunction!(native_unit_rotor, m)?)?;
    m.add_function(wrap_pyfunction!(native_unit_rotor_slice, m)?)?;
    m.add_function(wrap_pyfunction!(native_cubic_lagrange_weights, m)?)?;
    m.add_function(wrap_pyfunction!(native_givens_householder_mix_8ch, m)?)?;
    m.add_function(wrap_pyfunction!(native_mufdn_coercive_step, m)?)?;
    m.add_function(wrap_pyfunction!(native_mufdn_slice, m)?)?;
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
    Ok(())
}
