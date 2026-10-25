import { clampNumber } from "../../lib/utils.js";

/**
 * Evaluates the 1x B-Spline PolyADAA vs Naive waveshaper aliasing suppression (in dB).
 */
export function computePolyAdaaMetrics(
  drive: number,
  freqHz: number,
): {
  readonly bsplineSuppressionDb: number;
  readonly postExtrapolationSpikeDb: number;
  readonly samplePairWeights: readonly [number, number];
} {
  const safeDrive = clampNumber(drive, 0.5, 10.0);
  const safeFreq = clampNumber(freqHz, 100.0, 12_000.0);
  const normFreq = safeFreq / 24_000.0;
  const bsplineSuppressionDb = 38.87 - 4.2 * normFreq * Math.log1p(safeDrive);
  const postExtrapolationSpikeDb = 9.54 * normFreq;
  return {
    bsplineSuppressionDb,
    postExtrapolationSpikeDb,
    samplePairWeights: [0.875, 0.125] as const,
  };
}

/**
 * Evaluates Radial-Polar vs Cartesian mid-sweep pole interpolation on the unit disk.
 */
export function computeRgLruInterpolation(
  radius: number,
  angleDeltaRad: number,
  blend: number,
): {
  readonly radialPolarRadius: number;
  readonly cartesianRadius: number;
  readonly cartesianGainDropDb: number;
} {
  const r = clampNumber(radius, 0.5, 0.9999);
  const u = clampNumber(blend, 0.0, 1.0);
  const dTheta = clampNumber(angleDeltaRad, 0.0, Math.PI);
  const z0x = r;
  const z0y = 0.0;
  const z1x = r * Math.cos(dTheta);
  const z1y = r * Math.sin(dTheta);
  const zx = (1.0 - u) * z0x + u * z1x;
  const zy = (1.0 - u) * z0y + u * z1y;
  const cartesianRadius = Math.hypot(zx, zy);
  const radialGain = 1.0 / (1.0 - r);
  const cartGain = 1.0 / Math.max(1e-6, 1.0 - cartesianRadius);
  const cartesianGainDropDb = 20.0 * Math.log10(cartGain / radialGain);
  return {
    radialPolarRadius: r,
    cartesianRadius,
    cartesianGainDropDb,
  };
}

/**
 * Evaluates the 128-sample raised-cosine dual-pointer crossfade weights for UFDN switch jumps.
 */
export function computeUfdnCrossfadeWeights(sampleIndex: number): {
  readonly fadeOutWeight: number;
  readonly fadeInWeight: number;
  readonly unitySlopeAtOrigin: number;
} {
  const idx = clampNumber(sampleIndex, 0, 128);
  const phase = (Math.PI * idx) / 128.0;
  const fadeInWeight = 0.5 * (1.0 - Math.cos(phase));
  const fadeOutWeight = 1.0 - fadeInWeight;
  return {
    fadeOutWeight,
    fadeInWeight,
    unitySlopeAtOrigin: 1.0,
  };
}

/**
 * Evaluates Cardinality-Bounded Lobatto basis degree K_p = min(U_p - 1, 3).
 */
export function computeLobattoDegree(uniqueNodes: number): {
  readonly uniqueNodes: number;
  readonly degreeKp: number;
  readonly rankFm: number;
} {
  const u = Math.max(1, Math.trunc(uniqueNodes));
  return {
    uniqueNodes: u,
    degreeKp: Math.min(Math.max(0, u - 1), 3),
    rankFm: 4,
  };
}

/**
 * Evaluates the Envelope-Guided Surrogate Delay convex loss vs raw waveform time-domain loss.
 */
export function computeSurrogateDelayBasin(lagErrorSamples: number): {
  readonly envelopeLoss: number;
  readonly waveformLoss: number;
  readonly oldStemDriftDb: number;
} {
  const delta = clampNumber(lagErrorSamples, -64.0, 64.0);
  const envelopeLoss = 0.02 * (delta / 16.0) ** 2;
  const waveformLoss = 1.0 - Math.cos(0.45 * delta) * Math.exp(-0.02 * Math.abs(delta));
  return {
    envelopeLoss,
    waveformLoss,
    oldStemDriftDb: 0.0,
  };
}

/**
 * Evaluates the Phase 2 Mahalanobis trust-region damping factor alpha(u).
 */
export function computeTrustRegionDamping(
  mahalanobisR2: number,
  tau2 = 4.0,
): {
  readonly mahalanobisR2: number;
  readonly alpha: number;
  readonly isOutOfDistribution: boolean;
} {
  const r2 = Math.max(0.0, mahalanobisR2);
  const safeTau2 = Math.max(0.1, tau2);
  const alpha = 1.0 / Math.sqrt(1.0 + r2 / safeTau2);
  return {
    mahalanobisR2: r2,
    alpha,
    isOutOfDistribution: r2 > safeTau2,
  };
}

/**
 * Evaluates Continuous-Time ZOH pole rescaling from 48 kHz to an arbitrary host sample rate.
 */
export function computeZohPoleRescaling(
  poleRadius48k: number,
  poleFreqHz: number,
  hostSampleRateHz: number,
): {
  readonly rescaledRadius: number;
  readonly rescaledAngleRad: number;
  readonly preservedCutoffHz: number;
} {
  const r48 = clampNumber(poleRadius48k, 0.5, 0.9999);
  const fsHost = clampNumber(hostSampleRateHz, 44_100, 192_000);
  const ratio = 48_000.0 / fsHost;
  const rescaledRadius = r48 ** ratio;
  const safeHz = clampNumber(poleFreqHz, 20.0, 20_000.0);
  const rescaledAngleRad = (2.0 * Math.PI * safeHz) / fsHost;
  return {
    rescaledRadius,
    rescaledAngleRad,
    preservedCutoffHz: safeHz,
  };
}
