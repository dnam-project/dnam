//! Shared `f64`/`f32` bindings to Core 1, Core 2, Control Bus, and Math reference DSP kernels.

use dnam_engine::control::{
    eval_chebyshev_lfo_4harm_f64, eval_newton_lobatto_basis_f64, eval_rank4_fm_interaction_f64,
    eval_trust_region_softclip_f64,
};
use dnam_engine::core1::{poly_adaa_algebraic_f64, selective_rglru_step_f64};
use dnam_engine::core2::{
    cubic_lagrange_weights_f64, givens_householder_mix_8ch_f64, mufdn_coercive_step_f64,
    mufdn_slice_f64,
};
use dnam_engine::math::algebraic_unit_rotor;

/// Division-free algebraic `PolyADAA` segment average (`CL-01`).
#[inline]
pub fn poly_adaa_algebraic_core(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta)
}

/// Single-sample `PolyADAA` step returning `[y, next_u_prev, next_a_prev, next_m_dc]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_step_core(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
) -> [f64; 4] {
    let u_curr = ((x + b_conv) * (2.0 / 3.0)).clamp(-1.0, 1.0);
    let a_curr = poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta);
    let y_nl = 0.875 * a_curr + 0.125 * a_prev;
    let m_dc_curr = r_ac * m_dc + (1.0 - r_ac) * y_nl;
    let y = alpha * x + (y_nl - m_dc_curr);
    [y, u_curr, a_curr, m_dc_curr]
}

/// Processes a contiguous `f64` slice through `PolyADAA`, starting from `(u_prev, a_prev, m_dc)`
/// (or analytical DC resting equilibrium if `reset == true`) and returning `[next_u_prev, next_a_prev, next_m_dc]`.
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_scan_core(
    inp: &[f64],
    out: &mut [f64],
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    mut u_prev: f64,
    mut a_prev: f64,
    mut m_dc: f64,
    reset: bool,
) -> Result<[f64; 3], String> {
    if inp.len() != out.len() {
        return Err("[DNAM-VALIDATION] input and output must have identical length".to_string());
    }
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
    Ok([u_prev, a_prev, m_dc])
}

/// Single-sample Radial-Polar `SelectiveRgLru` step returning `[y, next_z_re, next_z_im]`.
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn selective_rglru_step_core(
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
) -> [f64; 3] {
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
    [y, z_re, z_im]
}

/// Processes a contiguous `f64` slice through `SelectiveRgLru`, starting from `(z_re, z_im)`
/// and returning `[next_z_re, next_z_im]`.
#[allow(clippy::too_many_arguments)]
pub fn selective_rglru_scan_core(
    inp: &[f64],
    out: &mut [f64],
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
) -> Result<[f64; 2], String> {
    if inp.len() != out.len() {
        return Err("[DNAM-VALIDATION] input and output must have identical length".to_string());
    }
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
    Ok([z_re, z_im])
}

/// 4-FMA + Algebraic Newton unit-circle rotor returning `[cos_theta, sin_theta]`.
#[inline]
pub fn unit_rotor_core(phi: f32) -> [f64; 2] {
    let (c, s) = algebraic_unit_rotor(phi);
    [f64::from(c), f64::from(s)]
}

/// Vectorized $3\text{ kHz}$ 4-FMA + Newton unit-circle rotor evaluation over a 1D `f64` slice into `2 * N` (`cos` row followed by `sin` row).
pub fn unit_rotor_slice_core(phi: &[f64], out: &mut [f64]) -> Result<(), String> {
    let n = phi.len();
    if out.len() != 2 * n {
        return Err("[DNAM-VALIDATION] unitRotor output length must be 2 * N".to_string());
    }
    let (cos_row, sin_row) = out.split_at_mut(n);
    for (i, &p_val) in phi.iter().enumerate() {
        let (c, s) = algebraic_unit_rotor(p_val as f32);
        cos_row[i] = f64::from(c);
        sin_row[i] = f64::from(s);
    }
    Ok(())
}

/// 4-tap Cubic Lagrange fractional delay weights (`CL-06`).
#[inline]
pub fn cubic_lagrange_weights_core(mu: f64) -> [f64; 4] {
    cubic_lagrange_weights_f64(mu)
}

/// 8-channel Orthogonal Givens + Householder mixing matrix `U(psi)` (`CL-08`).
pub fn givens_householder_mix_8ch_core(
    x: &[f64],
    cos_psi: f64,
    sin_psi: f64,
) -> Result<[f64; 8], String> {
    let arr: &[f64; 8] = x.try_into().map_err(|_| {
        format!(
            "[DNAM-VALIDATION] givensHouseholderMix requires 8 elements, got {}",
            x.len()
        )
    })?;
    Ok(givens_householder_mix_8ch_f64(arr, cos_psi, sin_psi))
}

/// Single-sample Coercive-Saturating Lyapunov loop step (`CL-09`..`CL-11`).
pub fn mufdn_coercive_step_core(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    mut dc_x_prev: f64,
    mut dc_y_prev: f64,
) -> [f64; 3] {
    let s_next = mufdn_coercive_step_f64(
        u_in,
        g_lin,
        sigma_knee,
        v_ceil,
        r_dc,
        &mut dc_x_prev,
        &mut dc_y_prev,
    );
    [s_next, dc_x_prev, dc_y_prev]
}

/// Processes a contiguous `f64` slice through a reference Coercive-Saturating `M-UFDN` loop.
#[allow(clippy::too_many_arguments)]
pub fn mufdn_slice_core(
    inp: &[f64],
    out: &mut [f64],
    delay_samples: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> Result<(), String> {
    if inp.len() != out.len() {
        return Err("[DNAM-VALIDATION] input and output must have identical length".to_string());
    }
    mufdn_slice_f64(inp, out, delay_samples, g_lin, sigma_knee, v_ceil, r_dc);
    Ok(())
}

/// Node-Anchored Newton/Lobatto basis evaluation (`CL-13`).
pub fn eval_lobatto_basis_core(u: f64, anchors: &[f64], degree: usize) -> Result<[f64; 3], String> {
    let anc: &[f64; 3] = anchors.try_into().map_err(|_| {
        format!(
            "[DNAM-VALIDATION] Lobatto anchors must have 3 elements, got {}",
            anchors.len()
        )
    })?;
    Ok(eval_newton_lobatto_basis_f64(u, *anc, degree))
}

/// Vectorized Node-Anchored Newton/Lobatto basis evaluation over a 1D `f64` slice into `3 * N`.
pub fn eval_lobatto_basis_slice_core(
    u: &[f64],
    out: &mut [f64],
    anchors: &[f64],
    degree: usize,
) -> Result<(), String> {
    let anc: &[f64; 3] = anchors.try_into().map_err(|_| {
        format!(
            "[DNAM-VALIDATION] Lobatto anchors must have 3 elements, got {}",
            anchors.len()
        )
    })?;
    let (out_rows, rem) = out.as_chunks_mut::<3>();
    if !rem.is_empty() || out_rows.len() != u.len() {
        return Err("[DNAM-VALIDATION] LobattoBasis output length must be 3 * N".to_string());
    }
    for (&u_val, row) in u.iter().zip(out_rows.iter_mut()) {
        *row = eval_newton_lobatto_basis_f64(u_val, *anc, degree);
    }
    Ok(())
}

/// Rank-4 Factorization Machine interaction evaluation (`CL-14`).
pub fn eval_rank4_fm_core(features: &[f64], v_factors_flat: &[f64]) -> Result<[f64; 4], String> {
    let d = features.len();
    if d == 0 || v_factors_flat.len() != d * 4 {
        return Err(format!(
            "[DNAM-VALIDATION] vFactors must have ({d} * 4) = {} elements, got {}",
            d * 4,
            v_factors_flat.len()
        ));
    }
    let (v_rows, _) = v_factors_flat.as_chunks::<4>();
    Ok(eval_rank4_fm_interaction_f64(features, v_rows))
}

/// 4-Harmonic Chebyshev LFO polynomial evaluation (`CL-16`).
#[inline]
pub fn eval_chebyshev_lfo_core(cos_phi: f64) -> [f64; 4] {
    eval_chebyshev_lfo_4harm_f64(cos_phi)
}

/// Vectorized 4-Harmonic Chebyshev LFO evaluation over a 1D `f64` slice into `4 * N`.
pub fn eval_chebyshev_lfo_slice_core(cos_phi: &[f64], out: &mut [f64]) -> Result<(), String> {
    let (out_rows, rem) = out.as_chunks_mut::<4>();
    if !rem.is_empty() || out_rows.len() != cos_phi.len() {
        return Err("[DNAM-VALIDATION] chebyshevLfo output length must be 4 * N".to_string());
    }
    for (&c_val, row) in cos_phi.iter().zip(out_rows.iter_mut()) {
        *row = eval_chebyshev_lfo_4harm_f64(c_val);
    }
    Ok(())
}

/// Cubic trust-region soft-clipper (`CL-15`).
#[inline]
pub fn eval_trust_region_softclip_core(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_f64(z, mh)
}

/// Vectorized cubic trust-region soft-clipper over a 1D `f64` slice.
pub fn eval_trust_region_softclip_slice_core(
    z: &[f64],
    out: &mut [f64],
    mh: f64,
) -> Result<(), String> {
    if z.len() != out.len() {
        return Err(
            "[DNAM-VALIDATION] trustRegionSoftclip input and output lengths must match".to_string(),
        );
    }
    for (&z_val, y_out) in z.iter().zip(out.iter_mut()) {
        *y_out = eval_trust_region_softclip_f64(z_val, mh);
    }
    Ok(())
}
