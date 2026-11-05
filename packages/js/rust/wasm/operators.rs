//! WASM TORA Reference DSP Kernels (`Op-T`, `Op-M`, `Op-N`, Control Bus, and Mixing)
//! (`ref[sdk.ts.dsp.reference-kernels]`).

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
use crate::wasm::abi::{set_error, slice_from_raw, slice_from_raw_mut};

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_catmull_rom_hermite(
    s_m1: f64,
    s_0: f64,
    s_1: f64,
    s_2: f64,
    mu: f64,
) -> f64 {
    catmull_rom_hermite_core(s_m1, s_0, s_1, s_2, mu)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_catmull_rom_hermite_slice(
    s_m1_ptr: *const f64,
    s_0_ptr: *const f64,
    s_1_ptr: *const f64,
    s_2_ptr: *const f64,
    mu_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let s_m1 = slice_from_raw(s_m1_ptr, len);
    let s_0 = slice_from_raw(s_0_ptr, len);
    let s_1 = slice_from_raw(s_1_ptr, len);
    let s_2 = slice_from_raw(s_2_ptr, len);
    let mu = slice_from_raw(mu_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    match catmull_rom_hermite_slice_core(s_m1, s_0, s_1, s_2, mu, out) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_op_t_coercive_step(
    u_in: f64,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
    dc_x_prev: f64,
    dc_y_prev: f64,
    out3_ptr: *mut f64,
) {
    let res = op_t_coercive_step_core(u_in, g_lin, sigma_knee, v_ceil, r_dc, dc_x_prev, dc_y_prev);
    // SAFETY: `out3_ptr` points to 3 f64 slots in WASM linear memory.
    let out = slice_from_raw_mut(out3_ptr, 3);
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_op_t_tank_slice(
    inp_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let inp = slice_from_raw(inp_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    match op_t_tank_slice_core(inp, out, g_lin, sigma_knee, v_ceil, r_dc) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_op_m_modal_step(
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
    out3_ptr: *mut f64,
) {
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
    // SAFETY: `out3_ptr` points to 3 f64 slots in WASM linear memory.
    let out = slice_from_raw_mut(out3_ptr, 3);
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_op_m_modal_slice(
    inp_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
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
    reset: u8,
    state2_ptr: *mut f64,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let inp = slice_from_raw(inp_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    match op_m_modal_slice_core(
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
        z_re,
        z_im,
        reset != 0,
    ) {
        Ok(st) => {
            let st_out = slice_from_raw_mut(state2_ptr, 2);
            st_out.copy_from_slice(&st);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unit_rotor(phi: f32, out2_ptr: *mut f64) {
    let res = unit_rotor_core(phi);
    // SAFETY: `out2_ptr` points to 2 f64 slots in WASM linear memory.
    let out = slice_from_raw_mut(out2_ptr, 2);
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unit_rotor_slice(
    phi_ptr: *const f64,
    len: usize,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let phi = slice_from_raw(phi_ptr, len);
    let out = slice_from_raw_mut(out_ptr, out_len);
    match unit_rotor_slice_core(phi, out) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_poly_adaa_algebraic(
    u_prev: f64,
    u_curr: f64,
    beta: f64,
    delta: f64,
) -> f64 {
    poly_adaa_algebraic_core(u_prev, u_curr, beta, delta)
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_poly_adaa_step(
    x: f64,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    out4_ptr: *mut f64,
) {
    let res = poly_adaa_step_core(x, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc);
    // SAFETY: `out4_ptr` points to 4 f64 slots in WASM linear memory.
    let out = slice_from_raw_mut(out4_ptr, 4);
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn dnam_wasm_poly_adaa_slice(
    inp_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    b_conv: f64,
    alpha: f64,
    beta: f64,
    delta: f64,
    r_ac: f64,
    u_prev: f64,
    a_prev: f64,
    m_dc: f64,
    reset: u8,
    state3_ptr: *mut f64,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let inp = slice_from_raw(inp_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    match poly_adaa_slice_core(
        inp,
        out,
        b_conv,
        alpha,
        beta,
        delta,
        r_ac,
        u_prev,
        a_prev,
        m_dc,
        reset != 0,
    ) {
        Ok(st) => {
            let st_out = slice_from_raw_mut(state3_ptr, 3);
            st_out.copy_from_slice(&st);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_lobatto_basis(
    u: f64,
    anchors_ptr: *const f64,
    anchors_len: usize,
    degree: usize,
    out3_ptr: *mut f64,
) -> i32 {
    // SAFETY: Caller passes valid f64 slices in WASM linear memory.
    let anchors = slice_from_raw(anchors_ptr, anchors_len);
    match eval_lobatto_basis_core(u, anchors, degree) {
        Ok(res) => {
            let out = slice_from_raw_mut(out3_ptr, 3);
            out.copy_from_slice(&res);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_lobatto_basis_slice(
    u_ptr: *const f64,
    u_len: usize,
    out_ptr: *mut f64,
    out_len: usize,
    anchors_ptr: *const f64,
    anchors_len: usize,
    degree: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let u = slice_from_raw(u_ptr, u_len);
    let out = slice_from_raw_mut(out_ptr, out_len);
    let anchors = slice_from_raw(anchors_ptr, anchors_len);
    match eval_lobatto_basis_slice_core(u, out, anchors, degree) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_rank4_fm(
    feat_ptr: *const f64,
    feat_len: usize,
    vf_ptr: *const f64,
    vf_len: usize,
    out4_ptr: *mut f64,
) -> i32 {
    // SAFETY: Caller passes valid f64 slices in WASM linear memory.
    let feat = slice_from_raw(feat_ptr, feat_len);
    let vf = slice_from_raw(vf_ptr, vf_len);
    match eval_rank4_fm_core(feat, vf) {
        Ok(res) => {
            let out = slice_from_raw_mut(out4_ptr, 4);
            out.copy_from_slice(&res);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_rank4_fm_slice(
    feat_ptr: *const f64,
    feat_len: usize,
    n_frames: usize,
    d_dim: usize,
    vf_ptr: *const f64,
    vf_len: usize,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let feat = slice_from_raw(feat_ptr, feat_len);
    let vf = slice_from_raw(vf_ptr, vf_len);
    let out = slice_from_raw_mut(out_ptr, out_len);
    match eval_rank4_fm_slice_core(feat, n_frames, d_dim, vf, out) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_chebyshev_lfo(cos_phi: f64, out4_ptr: *mut f64) {
    let res = eval_chebyshev_lfo_core(cos_phi);
    // SAFETY: `out4_ptr` points to 4 f64 slots in WASM linear memory.
    let out = slice_from_raw_mut(out4_ptr, 4);
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_chebyshev_lfo_slice(
    cos_phi_ptr: *const f64,
    len: usize,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let cos_phi = slice_from_raw(cos_phi_ptr, len);
    let out = slice_from_raw_mut(out_ptr, out_len);
    match eval_chebyshev_lfo_slice_core(cos_phi, out) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_trust_region_softclip(z: f64, mh: f64) -> f64 {
    eval_trust_region_softclip_core(z, mh)
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_trust_region_softclip_slice(
    z_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    mh: f64,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let z = slice_from_raw(z_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    match eval_trust_region_softclip_slice_core(z, out, mh) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_cubic_lagrange_weights(mu: f64, out4_ptr: *mut f64) {
    let res = cubic_lagrange_weights_core(mu);
    // SAFETY: `out4_ptr` points to 4 f64 slots in WASM linear memory.
    let out = slice_from_raw_mut(out4_ptr, 4);
    out.copy_from_slice(&res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_cubic_lagrange_weights_slice(
    mu_ptr: *const f64,
    len: usize,
    out_ptr: *mut f64,
    out_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let mu = slice_from_raw(mu_ptr, len);
    let out = slice_from_raw_mut(out_ptr, out_len);
    match cubic_lagrange_weights_slice_core(mu, out) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_8ch(
    x_ptr: *const f64,
    x_len: usize,
    cos_psi: f64,
    sin_psi: f64,
    out8_ptr: *mut f64,
) -> i32 {
    // SAFETY: Caller passes valid f64 slices in WASM linear memory.
    let x = slice_from_raw(x_ptr, x_len);
    match givens_householder_8ch_core(x, cos_psi, sin_psi) {
        Ok(res) => {
            let out = slice_from_raw_mut(out8_ptr, 8);
            out.copy_from_slice(&res);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_8ch_slice(
    x_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    cos_psi: f64,
    sin_psi: f64,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let x = slice_from_raw(x_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    match givens_householder_8ch_slice_core(x, out, cos_psi, sin_psi) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_16ch(
    x_ptr: *const f64,
    x_len: usize,
    cos_psi: f64,
    sin_psi: f64,
    cc_ptr: *const f64,
    cc_len: usize,
    cs_ptr: *const f64,
    cs_len: usize,
    out16_ptr: *mut f64,
) -> i32 {
    // SAFETY: Caller passes valid f64 slices in WASM linear memory.
    let x = slice_from_raw(x_ptr, x_len);
    let cc = slice_from_raw(cc_ptr, cc_len);
    let cs = slice_from_raw(cs_ptr, cs_len);
    match givens_householder_16ch_core(x, cos_psi, sin_psi, cc, cs) {
        Ok(res) => {
            let out = slice_from_raw_mut(out16_ptr, 16);
            out.copy_from_slice(&res);
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_16ch_slice(
    x_ptr: *const f64,
    out_ptr: *mut f64,
    len: usize,
    cos_psi: f64,
    sin_psi: f64,
    cc_ptr: *const f64,
    cc_len: usize,
    cs_ptr: *const f64,
    cs_len: usize,
) -> i32 {
    // SAFETY: Caller passes valid non-overlapping f64 slices in WASM linear memory.
    let x = slice_from_raw(x_ptr, len);
    let out = slice_from_raw_mut(out_ptr, len);
    let cc = slice_from_raw(cc_ptr, cc_len);
    let cs = slice_from_raw(cs_ptr, cs_len);
    match givens_householder_16ch_slice_core(x, out, cos_psi, sin_psi, cc, cs) {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}
