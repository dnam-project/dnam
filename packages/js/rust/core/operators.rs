//! Target-agnostic TORA Reference DSP Kernels (`Op-T`, `Op-M`, `Op-N`, Control Bus, and Mixing)
//! (`ref[sdk.ts.dsp.reference-kernels]`).

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

use crate::error::validation_err;

// ============================================================================
// 1. Op-T: Transport, Catmull-Rom & Coercive Tank Loop
// ============================================================================

#[inline]
pub fn catmull_rom_hermite_core(s_m1: f64, s_0: f64, s_1: f64, s_2: f64, mu: f64) -> f64 {
    catmull_rom_hermite_f64(s_m1, s_0, s_1, s_2, mu)
}

pub fn catmull_rom_hermite_slice_core(
    s_m1: &[f64],
    s_0: &[f64],
    s_1: &[f64],
    s_2: &[f64],
    mu: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    let n = s_0.len();
    if s_m1.len() != n || s_1.len() != n || s_2.len() != n || mu.len() != n || out.len() != n {
        return Err(validation_err(
            "catmullRomHermite slice inputs and output must have identical length",
        ));
    }
    for i in 0..n {
        out[i] = catmull_rom_hermite_f64(s_m1[i], s_0[i], s_1[i], s_2[i], mu[i]);
    }
    Ok(())
}

#[inline]
pub fn op_t_coercive_step_core(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    mut dc_x_prev: f64,
    mut dc_y_prev: f64,
) -> [f64; 3] {
    let s_next = op_t_coercive_step_f64(
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

pub fn op_t_tank_slice_core(
    inp: &[f64],
    out: &mut [f64],
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> Result<(), String> {
    if inp.len() != out.len() {
        return Err(validation_err(
            "input and output must have identical length",
        ));
    }
    op_t_tank_slice_f64(inp, out, g_lin, sigma_knee, v_ceil, r_dc);
    Ok(())
}

// ============================================================================
// 2. Op-M: Symplectic Modal Step & Contractive Rotor
// ============================================================================

#[inline]
#[allow(clippy::too_many_arguments)]
pub fn op_m_modal_step_core(
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
    [y, z_re, z_im]
}

#[allow(clippy::too_many_arguments)]
pub fn op_m_modal_slice_core(
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
        return Err(validation_err(
            "input and output must have identical length",
        ));
    }
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
    Ok([z_re, z_im])
}

#[inline]
pub fn unit_rotor_core(phi: f32) -> [f64; 2] {
    let (c, s) = algebraic_unit_rotor(phi);
    [f64::from(c), f64::from(s)]
}

pub fn unit_rotor_slice_core(phi: &[f64], out: &mut [f64]) -> Result<(), String> {
    let n = phi.len();
    if out.len() != 2 * n {
        return Err(validation_err("unitRotor output length must be 2 * N"));
    }
    let (cos_row, sin_row) = out.split_at_mut(n);
    for (i, &p_val) in phi.iter().enumerate() {
        let (c, s) = algebraic_unit_rotor(p_val as f32);
        cos_row[i] = f64::from(c);
        sin_row[i] = f64::from(s);
    }
    Ok(())
}

// ============================================================================
// 3. Op-N: Division-Free Algebraic PolyADAA
// ============================================================================

#[inline]
pub fn poly_adaa_algebraic_core(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    poly_adaa_algebraic_f64(u_prev, u_curr, beta, delta)
}

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

#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_slice_core(
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
        return Err(validation_err(
            "input and output must have identical length",
        ));
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

// ============================================================================
// 4. Control Bus & Mixing Reference Kernels
// ============================================================================

pub fn eval_lobatto_basis_core(u: f64, anchors: &[f64], degree: usize) -> Result<[f64; 3], String> {
    let anc: &[f64; 3] = anchors.try_into().map_err(|_| {
        validation_err(format!(
            "Lobatto anchors must have 3 elements, got {}",
            anchors.len()
        ))
    })?;
    Ok(eval_newton_lobatto_basis_f64(u, *anc, degree))
}

pub fn eval_lobatto_basis_slice_core(
    u: &[f64],
    out: &mut [f64],
    anchors: &[f64],
    degree: usize,
) -> Result<(), String> {
    let anc: &[f64; 3] = anchors.try_into().map_err(|_| {
        validation_err(format!(
            "Lobatto anchors must have 3 elements, got {}",
            anchors.len()
        ))
    })?;
    let (out_rows, rem) = out.as_chunks_mut::<3>();
    if !rem.is_empty() || out_rows.len() != u.len() {
        return Err(validation_err("lobattoBasis output length must be 3 * N"));
    }
    for (&u_val, row) in u.iter().zip(out_rows.iter_mut()) {
        *row = eval_newton_lobatto_basis_f64(u_val, *anc, degree);
    }
    Ok(())
}

pub fn eval_rank4_fm_core(features: &[f64], v_factors_flat: &[f64]) -> Result<[f64; 4], String> {
    let d = features.len();
    if d == 0 || v_factors_flat.len() != d * 4 {
        return Err(validation_err(format!(
            "vFactors must have ({d} * 4) = {} elements, got {}",
            d * 4,
            v_factors_flat.len()
        )));
    }
    let (v_rows, _) = v_factors_flat.as_chunks::<4>();
    Ok(eval_rank4_fm_interaction_f64(features, v_rows))
}

pub fn eval_rank4_fm_slice_core(
    features_flat: &[f64],
    n_frames: usize,
    d_dim: usize,
    v_factors_flat: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    if d_dim == 0 || features_flat.len() != n_frames * d_dim {
        return Err(validation_err(
            "featuresFlat length must equal nFrames * dDim",
        ));
    }
    if v_factors_flat.len() != d_dim * 4 {
        return Err(validation_err("vFactors length must equal dDim * 4"));
    }
    let (v_rows, _) = v_factors_flat.as_chunks::<4>();
    let (out_rows, rem) = out.as_chunks_mut::<4>();
    if !rem.is_empty() || out_rows.len() != n_frames {
        return Err(validation_err(
            "rank4Fm output length must equal nFrames * 4",
        ));
    }
    for (frame_idx, row) in out_rows.iter_mut().enumerate() {
        let feat_row = &features_flat[frame_idx * d_dim..(frame_idx + 1) * d_dim];
        *row = eval_rank4_fm_interaction_f64(feat_row, v_rows);
    }
    Ok(())
}

#[inline]
pub fn eval_chebyshev_lfo_core(cos_phi: f64) -> [f64; 4] {
    eval_chebyshev_lfo_4harm_f64(cos_phi)
}

pub fn eval_chebyshev_lfo_slice_core(cos_phi: &[f64], out: &mut [f64]) -> Result<(), String> {
    let (out_rows, rem) = out.as_chunks_mut::<4>();
    if !rem.is_empty() || out_rows.len() != cos_phi.len() {
        return Err(validation_err("chebyshevLfo output length must be 4 * N"));
    }
    for (&c_val, row) in cos_phi.iter().zip(out_rows.iter_mut()) {
        *row = eval_chebyshev_lfo_4harm_f64(c_val);
    }
    Ok(())
}

#[inline]
pub fn eval_trust_region_softclip_core(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_f64(z, mh)
}

pub fn eval_trust_region_softclip_slice_core(
    z: &[f64],
    out: &mut [f64],
    mh: f64,
) -> Result<(), String> {
    if z.len() != out.len() {
        return Err(validation_err(
            "trustRegionSoftclip input and output lengths must match",
        ));
    }
    for (&z_val, y_out) in z.iter().zip(out.iter_mut()) {
        *y_out = eval_trust_region_softclip_f64(z_val, mh);
    }
    Ok(())
}

#[inline]
pub fn cubic_lagrange_weights_core(mu: f64) -> [f64; 4] {
    cubic_lagrange_weights_f64(mu)
}

pub fn cubic_lagrange_weights_slice_core(mu: &[f64], out: &mut [f64]) -> Result<(), String> {
    let (out_rows, rem) = out.as_chunks_mut::<4>();
    if !rem.is_empty() || out_rows.len() != mu.len() {
        return Err(validation_err(
            "cubicLagrangeWeights output length must be 4 * N",
        ));
    }
    for (&m_val, row) in mu.iter().zip(out_rows.iter_mut()) {
        *row = cubic_lagrange_weights_f64(m_val);
    }
    Ok(())
}

pub fn givens_householder_8ch_core(
    x: &[f64],
    cos_psi: f64,
    sin_psi: f64,
) -> Result<[f64; 8], String> {
    let arr: &[f64; 8] = x.try_into().map_err(|_| {
        validation_err(format!(
            "givensHouseholder8ch requires 8 elements, got {}",
            x.len()
        ))
    })?;
    Ok(givens_householder_mix_8ch_f64(arr, cos_psi, sin_psi))
}

pub fn givens_householder_8ch_slice_core(
    x_flat: &[f64],
    out: &mut [f64],
    cos_psi: f64,
    sin_psi: f64,
) -> Result<(), String> {
    let (in_rows, in_rem) = x_flat.as_chunks::<8>();
    let (out_rows, out_rem) = out.as_chunks_mut::<8>();
    if !in_rem.is_empty() || !out_rem.is_empty() || in_rows.len() != out_rows.len() {
        return Err(validation_err(
            "givensHouseholder8chSlice input and output must be multiples of 8 of identical length",
        ));
    }
    for (inp, outp) in in_rows.iter().zip(out_rows.iter_mut()) {
        *outp = givens_householder_mix_8ch_f64(inp, cos_psi, sin_psi);
    }
    Ok(())
}

pub fn givens_householder_16ch_core(
    x: &[f64],
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: &[f64],
    cross_sin: &[f64],
) -> Result<[f64; 16], String> {
    let arr: &[f64; 16] = x.try_into().map_err(|_| {
        validation_err(format!(
            "givensHouseholder16ch requires 16 elements, got {}",
            x.len()
        ))
    })?;
    let cc: &[f64; 8] = cross_cos.try_into().map_err(|_| {
        validation_err(format!(
            "givensHouseholder16ch crossCos requires 8 elements, got {}",
            cross_cos.len()
        ))
    })?;
    let cs: &[f64; 8] = cross_sin.try_into().map_err(|_| {
        validation_err(format!(
            "givensHouseholder16ch crossSin requires 8 elements, got {}",
            cross_sin.len()
        ))
    })?;
    Ok(givens_householder_mix_16ch_f64(
        arr, cos_psi, sin_psi, cc, cs,
    ))
}

pub fn givens_householder_16ch_slice_core(
    x_flat: &[f64],
    out: &mut [f64],
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: &[f64],
    cross_sin: &[f64],
) -> Result<(), String> {
    let cc: &[f64; 8] = cross_cos.try_into().map_err(|_| {
        validation_err(format!(
            "givensHouseholder16chSlice crossCos requires 8 elements, got {}",
            cross_cos.len()
        ))
    })?;
    let cs: &[f64; 8] = cross_sin.try_into().map_err(|_| {
        validation_err(format!(
            "givensHouseholder16chSlice crossSin requires 8 elements, got {}",
            cross_sin.len()
        ))
    })?;
    let (in_rows, in_rem) = x_flat.as_chunks::<16>();
    let (out_rows, out_rem) = out.as_chunks_mut::<16>();
    if !in_rem.is_empty() || !out_rem.is_empty() || in_rows.len() != out_rows.len() {
        return Err(validation_err(
            "givensHouseholder16chSlice input and output must be multiples of 16 of identical length",
        ));
    }
    for (inp, outp) in in_rows.iter().zip(out_rows.iter_mut()) {
        *outp = givens_householder_mix_16ch_f64(inp, cos_psi, sin_psi, cc, cs);
    }
    Ok(())
}
