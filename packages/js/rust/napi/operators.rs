//! Node-API v10 TORA Reference DSP Kernels (`Op-T`, `Op-M`, `Op-N`, Control Bus, and Mixing)
//! (`ref[sdk.ts.dsp.reference-kernels]`).

use napi::bindgen_prelude::{Float64Array, Result};
use napi_derive::napi;

use crate::core::operators::{
    catmull_rom_hermite_core, catmull_rom_hermite_slice_core, cubic_lagrange_weights_core,
    cubic_lagrange_weights_slice_core, eval_chebyshev_lfo_core, eval_chebyshev_lfo_slice_core,
    eval_lobatto_basis_core, eval_lobatto_basis_slice_core, eval_rank4_fm_core,
    eval_rank4_fm_slice_core, eval_trust_region_softclip_core,
    eval_trust_region_softclip_slice_core, givens_householder_8ch_core,
    givens_householder_8ch_slice_core, givens_householder_16ch_core,
    givens_householder_16ch_slice_core, op_m_modal_slice_core, op_m_modal_step_core,
    op_t_coercive_step_core, op_t_tank_slice_core, poly_adaa_algebraic_core, poly_adaa_slice_core,
    poly_adaa_step_core, unit_rotor_core, unit_rotor_slice_core,
};
use crate::napi::{f64_slice_mut, napi_err};

#[napi]
pub fn catmull_rom_hermite_native(s_m1: f64, s_0: f64, s_1: f64, s_2: f64, mu: f64) -> f64 {
    catmull_rom_hermite_core(s_m1, s_0, s_1, s_2, mu)
}

#[napi]
pub fn catmull_rom_hermite_slice_native(
    s_m1: Float64Array,
    s_0: Float64Array,
    s_1: Float64Array,
    s_2: Float64Array,
    mu: Float64Array,
    mut out: Float64Array,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    catmull_rom_hermite_slice_core(&s_m1, &s_0, &s_1, &s_2, &mu, o).map_err(napi_err)
}

#[napi]
pub fn op_t_coercive_step_native(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: f64,
    dc_y_prev: f64,
) -> Float64Array {
    let res = op_t_coercive_step_core(u_in, g_lin, sigma_knee, v_ceil, r_dc, dc_x_prev, dc_y_prev);
    Float64Array::new(res.to_vec())
}

#[napi]
pub fn op_t_tank_slice_native(
    inp: Float64Array,
    mut out: Float64Array,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    op_t_tank_slice_core(&inp, o, g_lin, sigma_knee, v_ceil, r_dc).map_err(napi_err)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn op_m_modal_step_native(
    x: f64,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: f64,
    z_im: f64,
) -> Float64Array {
    let res = op_m_modal_step_core(
        x,
        r_low,
        dr_span,
        w_sel,
        b_gamma_norm,
        c_real,
        c_imag,
        u_rot_cos,
        u_rot_sin,
        z_re,
        z_im,
    );
    Float64Array::new(res.to_vec())
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn op_m_modal_slice_native(
    inp: Float64Array,
    mut out: Float64Array,
    r_low: f64,
    dr_span: f64,
    w_sel: f64,
    b_gamma_norm: f64,
    c_real: f64,
    c_imag: f64,
    u_rot_cos: f64,
    u_rot_sin: f64,
    z_re: f64,
    z_im: f64,
    reset: bool,
) -> Result<Float64Array> {
    let o = f64_slice_mut(&mut out);
    let state = op_m_modal_slice_core(
        &inp,
        o,
        r_low,
        dr_span,
        w_sel,
        b_gamma_norm,
        c_real,
        c_imag,
        u_rot_cos,
        u_rot_sin,
        z_re,
        z_im,
        reset,
    )
    .map_err(napi_err)?;
    Ok(Float64Array::new(state.to_vec()))
}

#[napi]
pub fn unit_rotor_native(phi: f64) -> Float64Array {
    let res = unit_rotor_core(phi as f32);
    Float64Array::new(res.to_vec())
}

#[napi]
pub fn unit_rotor_slice_native(phi: Float64Array, mut out: Float64Array) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    unit_rotor_slice_core(&phi, o).map_err(napi_err)
}

#[napi]
pub fn poly_adaa_algebraic_native(u_prev: f64, u_curr: f64, beta: f64, delta: f64) -> f64 {
    poly_adaa_algebraic_core(u_prev, u_curr, beta, delta)
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_step_native(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
) -> Float64Array {
    let res = poly_adaa_step_core(x, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc);
    Float64Array::new(res.to_vec())
}

#[napi]
#[allow(clippy::too_many_arguments)]
pub fn poly_adaa_slice_native(
    inp: Float64Array,
    mut out: Float64Array,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    reset: bool,
) -> Result<Float64Array> {
    let o = f64_slice_mut(&mut out);
    let state = poly_adaa_slice_core(
        &inp, o, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc, reset,
    )
    .map_err(napi_err)?;
    Ok(Float64Array::new(state.to_vec()))
}

#[napi]
pub fn eval_lobatto_basis_native(
    u: f64,
    anchors: Float64Array,
    degree: u32,
) -> Result<Float64Array> {
    let res = eval_lobatto_basis_core(u, &anchors, degree as usize).map_err(napi_err)?;
    Ok(Float64Array::new(res.to_vec()))
}

#[napi]
pub fn eval_lobatto_basis_slice_native(
    u: Float64Array,
    mut out: Float64Array,
    anchors: Float64Array,
    degree: u32,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    eval_lobatto_basis_slice_core(&u, o, &anchors, degree as usize).map_err(napi_err)
}

#[napi]
pub fn eval_rank4_fm_native(
    features: Float64Array,
    v_factors_flat: Float64Array,
) -> Result<Float64Array> {
    let res = eval_rank4_fm_core(&features, &v_factors_flat).map_err(napi_err)?;
    Ok(Float64Array::new(res.to_vec()))
}

#[napi]
pub fn eval_rank4_fm_slice_native(
    features_flat: Float64Array,
    n_frames: u32,
    d_dim: u32,
    v_factors_flat: Float64Array,
    mut out: Float64Array,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    eval_rank4_fm_slice_core(
        &features_flat,
        n_frames as usize,
        d_dim as usize,
        &v_factors_flat,
        o,
    )
    .map_err(napi_err)
}

#[napi]
pub fn eval_chebyshev_lfo_native(cos_phi: f64) -> Float64Array {
    let res = eval_chebyshev_lfo_core(cos_phi);
    Float64Array::new(res.to_vec())
}

#[napi]
pub fn eval_chebyshev_lfo_slice_native(cos_phi: Float64Array, mut out: Float64Array) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    eval_chebyshev_lfo_slice_core(&cos_phi, o).map_err(napi_err)
}

#[napi]
pub fn eval_trust_region_softclip_native(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_core(z, mh)
}

#[napi]
pub fn eval_trust_region_softclip_slice_native(
    z: Float64Array,
    mut out: Float64Array,
    mh: f64,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    eval_trust_region_softclip_slice_core(&z, o, mh).map_err(napi_err)
}

#[napi]
pub fn cubic_lagrange_weights_native(mu: f64) -> Float64Array {
    let res = cubic_lagrange_weights_core(mu);
    Float64Array::new(res.to_vec())
}

#[napi]
pub fn cubic_lagrange_weights_slice_native(mu: Float64Array, mut out: Float64Array) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    cubic_lagrange_weights_slice_core(&mu, o).map_err(napi_err)
}

#[napi]
pub fn givens_householder_8ch_native(
    x: Float64Array,
    cos_psi: f64,
    sin_psi: f64,
) -> Result<Float64Array> {
    let res = givens_householder_8ch_core(&x, cos_psi, sin_psi).map_err(napi_err)?;
    Ok(Float64Array::new(res.to_vec()))
}

#[napi]
pub fn givens_householder_8ch_slice_native(
    x_flat: Float64Array,
    mut out: Float64Array,
    cos_psi: f64,
    sin_psi: f64,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    givens_householder_8ch_slice_core(&x_flat, o, cos_psi, sin_psi).map_err(napi_err)
}

#[napi]
pub fn givens_householder_16ch_native(
    x: Float64Array,
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: Float64Array,
    cross_sin: Float64Array,
) -> Result<Float64Array> {
    let res = givens_householder_16ch_core(&x, cos_psi, sin_psi, &cross_cos, &cross_sin)
        .map_err(napi_err)?;
    Ok(Float64Array::new(res.to_vec()))
}

#[napi]
pub fn givens_householder_16ch_slice_native(
    x_flat: Float64Array,
    mut out: Float64Array,
    cos_psi: f64,
    sin_psi: f64,
    cross_cos: Float64Array,
    cross_sin: Float64Array,
) -> Result<()> {
    let o = f64_slice_mut(&mut out);
    givens_householder_16ch_slice_core(&x_flat, o, cos_psi, sin_psi, &cross_cos, &cross_sin)
        .map_err(napi_err)
}
