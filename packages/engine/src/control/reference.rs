//! High-precision `f64` reference DSP kernels for the $3\text{ kHz}$ Parametric Control Bus (`CL-13`..`CL-16`).

/// Reference `f64` Node-Anchored Newton/Lobatto basis evaluation (`CL-13`):
/// $$\psi_1(u) = u - u_1, \quad \psi_2(u) = (u - u_1)(u - u_2), \quad \psi_3(u) = (u - u_1)(u - u_2)(u - u_3)$$
#[inline]
pub fn eval_newton_lobatto_basis_f64(u: f64, anchors: [f64; 3], degree: usize) -> [f64; 3] {
    if degree == 0 {
        return [0.0; 3];
    }
    let d1 = u - anchors[0];
    let p1 = d1;
    let p2 = if degree >= 2 {
        d1 * (u - anchors[1])
    } else {
        0.0
    };
    let p3 = if degree >= 3 {
        p2 * (u - anchors[2])
    } else {
        0.0
    };
    [p1, p2, p3]
}

/// Reference `f64` Rank-4 Factorization Machine interaction evaluation (`CL-14`):
/// $$\Delta\theta_{\text{FM}}(\mathbf{x}) = \frac{1}{2} \sum_{r=1}^R \left[\left(\sum_{i=1}^D v_{i,r} x_i\right)^2 - \sum_{i=1}^D v_{i,r}^2 x_i^2\right]$$
pub fn eval_rank4_fm_interaction_f64(features: &[f64], v_factors: &[[f64; 4]]) -> [f64; 4] {
    let mut sum_vx = [0.0_f64; 4];
    let mut sum_v2x2 = [0.0_f64; 4];
    let n = features.len().min(v_factors.len());
    for i in 0..n {
        let x = features[i];
        for r in 0..4 {
            let vx = v_factors[i][r] * x;
            sum_vx[r] += vx;
            sum_v2x2[r] += vx * vx;
        }
    }
    let mut out = [0.0_f64; 4];
    for r in 0..4 {
        out[r] = 0.5 * (sum_vx[r] * sum_vx[r] - sum_v2x2[r]);
    }
    out
}

/// Reference `f64` 4-Harmonic Chebyshev LFO polynomial evaluation ($T_1..T_4(\cos\phi)$, `CL-16`).
#[inline]
pub fn eval_chebyshev_lfo_4harm_f64(cos_phi: f64) -> [f64; 4] {
    let t1 = cos_phi;
    let two_x = 2.0 * t1;
    let t2 = two_x * t1 - 1.0;
    let t3 = two_x * t2 - t1;
    let t4 = two_x * t3 - t2;
    [t1, t2, t3, t4]
}

/// Reference `f64` cubic trust-region soft-clipper (`CL-15`):
/// $$z_{\text{safe}} = M_h \cdot S_3(z / M_h), \qquad S_3(v) = \operatorname{clamp}(v, -1, 1)\big(1.5 - 0.5\operatorname{clamp}(v, -1, 1)^2\big)$$
#[inline]
pub fn eval_trust_region_softclip_f64(z: f64, mh: f64) -> f64 {
    let v = (z / mh).clamp(-1.0, 1.0);
    mh * v * (1.5 - 0.5 * v * v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::control::ControlClampDescriptor;

    #[test]
    fn test_reference_lobatto_fm_chebyshev_and_trust_region() {
        for deg in 0..=3 {
            let mut d = ControlClampDescriptor::empty();
            d.ctrl_kind = 0;
            d.lobatto_degree = deg;
            d.anchor_nodes_norm = [0.0, -0.5, 0.5, 0.0];
            let b32 = d.eval_lobatto_basis(0.25);
            let b64 = eval_newton_lobatto_basis_f64(0.25, [0.0, -0.5, 0.5], deg as usize);
            assert!((b32[0] as f64 - b64[0]).abs() < 1e-6);
        }

        let x = [0.5_f64, -0.25, 0.75, 1.0];
        let v = [
            [0.2, -0.1, 0.4, 0.3],
            [-0.3, 0.5, 0.1, -0.2],
            [0.4, 0.2, -0.5, 0.1],
            [0.1, -0.4, 0.3, 0.5],
        ];
        let fm = eval_rank4_fm_interaction_f64(&x, &v);
        for r in 0..4 {
            let mut pairwise = 0.0;
            for i in 0..4 {
                for j in (i + 1)..4 {
                    pairwise += v[i][r] * v[j][r] * x[i] * x[j];
                }
            }
            assert!((fm[r] - pairwise).abs() < 1e-15);
        }

        let cheb = eval_chebyshev_lfo_4harm_f64(0.5);
        assert!((cheb[0] - 0.5).abs() < 1e-15);
        assert!((cheb[1] - (-0.5)).abs() < 1e-15);

        let clipped = eval_trust_region_softclip_f64(10.0, 2.5);
        assert!((clipped - 2.5).abs() < 1e-15);
    }
}
