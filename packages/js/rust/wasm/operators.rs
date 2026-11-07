//! WASM TORA Reference DSP Kernels (`Op-T`, `Op-M`, `Op-N`, Control Bus, and Mixing)
//! (`ref[sdk.ts.dsp.reference-kernels]`, `ref[sdk.ts.runtime.zero-unsafe]`).

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
use crate::wasm::abi::{read_f64_vec, set_error, with_f64_mut, write_f64_slice};

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
    s_m1_ptr: usize,
    s_0_ptr: usize,
    s_1_ptr: usize,
    s_2_ptr: usize,
    mu_ptr: usize,
    out_ptr: usize,
    len: usize,
) -> i32 {
    let s_m1 = read_f64_vec(s_m1_ptr, len);
    let s_0 = read_f64_vec(s_0_ptr, len);
    let s_1 = read_f64_vec(s_1_ptr, len);
    let s_2 = read_f64_vec(s_2_ptr, len);
    let mu = read_f64_vec(mu_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        catmull_rom_hermite_slice_core(&s_m1, &s_0, &s_1, &s_2, &mu, out)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
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
    out3_ptr: usize,
) {
    let res = op_t_coercive_step_core(u_in, g_lin, sigma_knee, v_ceil, r_dc, dc_x_prev, dc_y_prev);
    let _ = write_f64_slice(out3_ptr, &res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_op_t_tank_slice(
    inp_ptr: usize,
    out_ptr: usize,
    len: usize,
    g_lin: f64,
    sigma_knee: f64,
    v_ceil: f64,
    r_dc: f64,
) -> i32 {
    let inp = read_f64_vec(inp_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        op_t_tank_slice_core(&inp, out, g_lin, sigma_knee, v_ceil, r_dc)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
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
    out3_ptr: usize,
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
    let _ = write_f64_slice(out3_ptr, &res);
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_op_m_modal_slice(
    inp_ptr: usize,
    out_ptr: usize,
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
    state2_ptr: usize,
) -> i32 {
    let inp = read_f64_vec(inp_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        op_m_modal_slice_core(
            &inp,
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
        )
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(st) => {
            if !write_f64_slice(state2_ptr, &st) {
                return set_error("[DNAM-VALIDATION] invalid state2_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unit_rotor(phi: f32, out2_ptr: usize) {
    let res = unit_rotor_core(phi);
    let _ = write_f64_slice(out2_ptr, &res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_unit_rotor_slice(
    phi_ptr: usize,
    len: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let phi = read_f64_vec(phi_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| unit_rotor_slice_core(&phi, out)) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
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

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
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
    out4_ptr: usize,
) {
    let res = poly_adaa_step_core(x, b_conv, alpha, beta, delta, r_ac, u_prev, a_prev, m_dc);
    let _ = write_f64_slice(out4_ptr, &res);
}

#[allow(clippy::too_many_arguments)]
#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_poly_adaa_slice(
    inp_ptr: usize,
    out_ptr: usize,
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
    state3_ptr: usize,
) -> i32 {
    let inp = read_f64_vec(inp_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        poly_adaa_slice_core(
            &inp,
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
        )
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(st) => {
            if !write_f64_slice(state3_ptr, &st) {
                return set_error("[DNAM-VALIDATION] invalid state3_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_lobatto_basis(
    u: f64,
    anchors_ptr: usize,
    anchors_len: usize,
    degree: usize,
    out3_ptr: usize,
) -> i32 {
    let anchors = read_f64_vec(anchors_ptr, anchors_len);
    match eval_lobatto_basis_core(u, &anchors, degree) {
        Ok(res) => {
            if !write_f64_slice(out3_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out3_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_lobatto_basis_slice(
    u_ptr: usize,
    u_len: usize,
    out_ptr: usize,
    out_len: usize,
    anchors_ptr: usize,
    anchors_len: usize,
    degree: usize,
) -> i32 {
    let u = read_f64_vec(u_ptr, u_len);
    let anchors = read_f64_vec(anchors_ptr, anchors_len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| {
        eval_lobatto_basis_slice_core(&u, out, &anchors, degree)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_rank4_fm(
    feat_ptr: usize,
    feat_len: usize,
    vf_ptr: usize,
    vf_len: usize,
    out4_ptr: usize,
) -> i32 {
    let feat = read_f64_vec(feat_ptr, feat_len);
    let vf = read_f64_vec(vf_ptr, vf_len);
    match eval_rank4_fm_core(&feat, &vf) {
        Ok(res) => {
            if !write_f64_slice(out4_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out4_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_rank4_fm_slice(
    feat_ptr: usize,
    feat_len: usize,
    n_frames: usize,
    d_dim: usize,
    vf_ptr: usize,
    vf_len: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let feat = read_f64_vec(feat_ptr, feat_len);
    let vf = read_f64_vec(vf_ptr, vf_len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| {
        eval_rank4_fm_slice_core(&feat, n_frames, d_dim, &vf, out)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_chebyshev_lfo(cos_phi: f64, out4_ptr: usize) {
    let res = eval_chebyshev_lfo_core(cos_phi);
    let _ = write_f64_slice(out4_ptr, &res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_eval_chebyshev_lfo_slice(
    cos_phi_ptr: usize,
    len: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let cos_phi = read_f64_vec(cos_phi_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| {
        eval_chebyshev_lfo_slice_core(&cos_phi, out)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
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
    z_ptr: usize,
    out_ptr: usize,
    len: usize,
    mh: f64,
) -> i32 {
    let z = read_f64_vec(z_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        eval_trust_region_softclip_slice_core(&z, out, mh)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_cubic_lagrange_weights(mu: f64, out4_ptr: usize) {
    let res = cubic_lagrange_weights_core(mu);
    let _ = write_f64_slice(out4_ptr, &res);
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_cubic_lagrange_weights_slice(
    mu_ptr: usize,
    len: usize,
    out_ptr: usize,
    out_len: usize,
) -> i32 {
    let mu = read_f64_vec(mu_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, out_len, |out| {
        cubic_lagrange_weights_slice_core(&mu, out)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_8ch(
    x_ptr: usize,
    x_len: usize,
    cos_psi: f64,
    sin_psi: f64,
    out8_ptr: usize,
) -> i32 {
    let x = read_f64_vec(x_ptr, x_len);
    match givens_householder_8ch_core(&x, cos_psi, sin_psi) {
        Ok(res) => {
            if !write_f64_slice(out8_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out8_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_8ch_slice(
    x_ptr: usize,
    out_ptr: usize,
    len: usize,
    cos_psi: f64,
    sin_psi: f64,
) -> i32 {
    let x = read_f64_vec(x_ptr, len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        givens_householder_8ch_slice_core(&x, out, cos_psi, sin_psi)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_16ch(
    x_ptr: usize,
    x_len: usize,
    cos_psi: f64,
    sin_psi: f64,
    cc_ptr: usize,
    cc_len: usize,
    cs_ptr: usize,
    cs_len: usize,
    out16_ptr: usize,
) -> i32 {
    let x = read_f64_vec(x_ptr, x_len);
    let cc = read_f64_vec(cc_ptr, cc_len);
    let cs = read_f64_vec(cs_ptr, cs_len);
    match givens_householder_16ch_core(&x, cos_psi, sin_psi, &cc, &cs) {
        Ok(res) => {
            if !write_f64_slice(out16_ptr, &res) {
                return set_error("[DNAM-VALIDATION] invalid out16_ptr".to_string());
            }
            0
        }
        Err(msg) => set_error(msg),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn dnam_wasm_givens_householder_16ch_slice(
    x_ptr: usize,
    out_ptr: usize,
    len: usize,
    cos_psi: f64,
    sin_psi: f64,
    cc_ptr: usize,
    cc_len: usize,
    cs_ptr: usize,
    cs_len: usize,
) -> i32 {
    let x = read_f64_vec(x_ptr, len);
    let cc = read_f64_vec(cc_ptr, cc_len);
    let cs = read_f64_vec(cs_ptr, cs_len);
    let Some(res) = with_f64_mut(out_ptr, len, |out| {
        givens_householder_16ch_slice_core(&x, out, cos_psi, sin_psi, &cc, &cs)
    }) else {
        return set_error("[DNAM-VALIDATION] invalid out_ptr".to_string());
    };
    match res {
        Ok(()) => 0,
        Err(msg) => set_error(msg),
    }
}
