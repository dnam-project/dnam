/**
 * Reference `f64` DSP Kernels for Core 1 (`PolyAdaa`, `SelectiveRgLru`, `unitRotor`)
 * and Core 2 (`ModulatedUfdn`, `lagrangeWeights`, `givensHouseholderMix`).
 */

import { type BackendChoice, resolveDriver } from "./drivers/index.js";
import { validateStandardSchema } from "./schema/base.js";
import {
  type ModulatedUfdnParams,
  ModulatedUfdnParamsSchema,
  type PolyAdaaParams,
  PolyAdaaParamsSchema,
  type SelectiveRgLruParams,
  SelectiveRgLruParamsSchema,
} from "./schema/runtime.js";

export interface DspScanOptions {
  out?: Float64Array | undefined;
  reset?: boolean | undefined;
  backend?: BackendChoice | undefined;
}

/**
 * Core 1A: 1x B-Spline `PolyADAA` with Division-Free Algebraic Factorization (`CL-01`)
 * and $5\text{ Hz}$ AC-Coupled Zero-DC Invariant (`CL-02`).
 */
export class PolyAdaa {
  public readonly bConv: number;
  public readonly alpha: number;
  public readonly beta: number;
  public readonly delta: number;
  public readonly rAc: number;
  private readonly backend?: BackendChoice | undefined;
  public uPrev = 0.0;
  public aPrev = 0.0;
  public mDc = 0.0;

  public constructor(params: PolyAdaaParams = {}, backend?: BackendChoice) {
    const validated = validateStandardSchema(PolyAdaaParamsSchema, params, "PolyAdaaParams");
    this.bConv = validated.bConv ?? 0.0;
    this.alpha = validated.alpha ?? 1.0;
    this.beta = validated.beta ?? 0.5;
    this.delta = validated.delta ?? 0.1;
    this.rAc = validated.rAc ?? 0.9993;
    this.backend = backend;
    this.reset();
  }

  /**
   * Resets internal state $(u_{\text{prev}}, a_{\text{prev}}, m_{\text{dc}})$ to the exact
   * analytical DC resting equilibrium (`CL-02`) so Sample 0 produces `0.00000000`.
   */
  public reset(): void {
    const uRest = Math.min(1.0, Math.max(-1.0, this.bConv * (2.0 / 3.0)));
    const u2 = uRest * uRest;
    const gRest = this.beta * uRest * (1.5 - 0.5 * u2) + this.delta * (u2 - u2 * u2);
    this.uPrev = uRest;
    this.aPrev = gRest;
    this.mDc = gRest;
  }

  /**
   * Processes a single `f64` sample and advances internal state.
   */
  public step(x: number): number {
    const driver = resolveDriver(this.backend);
    const [y, nextU, nextA, nextM] = driver.polyAdaaStep(
      x,
      this.bConv,
      this.alpha,
      this.beta,
      this.delta,
      this.rAc,
      this.uPrev,
      this.aPrev,
      this.mDc,
    );
    this.uPrev = nextU;
    this.aPrev = nextA;
    this.mDc = nextM;
    return y;
  }

  /**
   * Processes a contiguous `Float64Array` slice, either continuing from current state
   * (`reset: false` default) or resetting to analytical DC equilibrium (`reset: true`).
   */
  public process(input: Float64Array, options: DspScanOptions = {}): Float64Array {
    const driver = resolveDriver(options.backend ?? this.backend);
    const out = options.out ?? new Float64Array(input.length);
    const reset = options.reset ?? false;
    const [nextU, nextA, nextM] = driver.polyAdaaScan(
      input,
      out,
      this.bConv,
      this.alpha,
      this.beta,
      this.delta,
      this.rAc,
      this.uPrev,
      this.aPrev,
      this.mDc,
      reset,
    );
    this.uPrev = nextU;
    this.aPrev = nextA;
    this.mDc = nextM;
    return out;
  }

  /**
   * Division-free algebraic `PolyADAA` segment average $A(u_0, u_1)$ (`CL-01`).
   */
  public static algebraic(
    uPrev: number,
    uCurr: number,
    beta = 0.5,
    delta = 0.1,
    backend?: BackendChoice,
  ): number {
    const driver = resolveDriver(backend);
    return driver.polyAdaaAlgebraic(uPrev, uCurr, beta, delta);
  }

  /**
   * Stateless convenience scan starting from analytical DC resting equilibrium.
   */
  public static scan(
    input: Float64Array,
    params: PolyAdaaParams = {},
    options: DspScanOptions = {},
  ): Float64Array {
    const adaa = new PolyAdaa(params, options.backend);
    return adaa.process(input, { ...options, reset: true });
  }
}

/**
 * Core 1B: Radial-Polar Selective Complex `RG-LRU` (`CL-03`..`CL-05`).
 */
export class SelectiveRgLru {
  public readonly rLow: number;
  public readonly drSpan: number;
  public readonly wSel: number;
  public readonly bGammaNorm: number;
  public readonly cReal: number;
  public readonly cImag: number;
  public readonly uRotCos: number;
  public readonly uRotSin: number;
  private readonly backend?: BackendChoice | undefined;
  public zRe = 0.0;
  public zIm = 0.0;

  public constructor(params: SelectiveRgLruParams = {}, backend?: BackendChoice) {
    const validated = validateStandardSchema(
      SelectiveRgLruParamsSchema,
      params,
      "SelectiveRgLruParams",
    );
    this.rLow = validated.rLow ?? 0.9;
    this.drSpan = validated.drSpan ?? 0.05;
    this.wSel = validated.wSel ?? 0.5;
    this.bGammaNorm = validated.bGammaNorm ?? 0.25;
    this.cReal = validated.cReal ?? 1.0;
    this.cImag = validated.cImag ?? 0.0;
    this.uRotCos = validated.uRotCos ?? 0.98;
    this.uRotSin = validated.uRotSin ?? 0.198997;
    this.backend = backend;
  }

  public reset(): void {
    this.zRe = 0.0;
    this.zIm = 0.0;
  }

  public step(x: number): number {
    const driver = resolveDriver(this.backend);
    const [y, nextRe, nextIm] = driver.selectiveRgLruStep(
      x,
      this.rLow,
      this.drSpan,
      this.wSel,
      this.bGammaNorm,
      this.cReal,
      this.cImag,
      this.uRotCos,
      this.uRotSin,
      this.zRe,
      this.zIm,
    );
    this.zRe = nextRe;
    this.zIm = nextIm;
    return y;
  }

  public process(input: Float64Array, options: DspScanOptions = {}): Float64Array {
    const driver = resolveDriver(options.backend ?? this.backend);
    const out = options.out ?? new Float64Array(input.length);
    const reset = options.reset ?? false;
    const [nextRe, nextIm] = driver.selectiveRgLruScan(
      input,
      out,
      this.rLow,
      this.drSpan,
      this.wSel,
      this.bGammaNorm,
      this.cReal,
      this.cImag,
      this.uRotCos,
      this.uRotSin,
      this.zRe,
      this.zIm,
      reset,
    );
    this.zRe = nextRe;
    this.zIm = nextIm;
    return out;
  }

  public static scan(
    input: Float64Array,
    params: SelectiveRgLruParams = {},
    options: DspScanOptions = {},
  ): Float64Array {
    const rglru = new SelectiveRgLru(params, options.backend);
    return rglru.process(input, { ...options, reset: true });
  }
}

/**
 * Core 2: Diffusive Modulated Unitary Feedback Delay Network (`M-UFDN`) Coercive-Saturating
 * Lyapunov loop (`CL-09`..`CL-12`).
 */
export class ModulatedUfdn {
  public readonly delaySamples: number;
  public readonly gLin: number;
  public readonly sigmaKnee: number;
  public readonly vCeil: number;
  public readonly rDc: number;
  private readonly backend?: BackendChoice | undefined;
  public dcXPrev = 0.0;
  public dcYPrev = 0.0;

  public constructor(params: ModulatedUfdnParams = {}, backend?: BackendChoice) {
    const validated = validateStandardSchema(
      ModulatedUfdnParamsSchema,
      params,
      "ModulatedUfdnParams",
    );
    this.delaySamples = validated.delaySamples ?? 64;
    this.gLin = validated.gLin ?? 0.85;
    this.sigmaKnee = validated.sigmaKnee ?? 0.25;
    this.vCeil = validated.vCeil ?? 0.985;
    this.rDc = validated.rDc ?? 0.995;
    this.backend = backend;
  }

  public reset(): void {
    this.dcXPrev = 0.0;
    this.dcYPrev = 0.0;
  }

  /**
   * Evaluates a single Coercive-Saturating + DC-blocker loop step (`CL-09`..`CL-11`).
   */
  public stepCoercive(uIn: number): number {
    const driver = resolveDriver(this.backend);
    const [sNext, nextX, nextY] = driver.mufdnCoerciveStep(
      uIn,
      this.gLin,
      this.sigmaKnee,
      this.vCeil,
      this.rDc,
      this.dcXPrev,
      this.dcYPrev,
    );
    this.dcXPrev = nextX;
    this.dcYPrev = nextY;
    return sNext;
  }

  public process(input: Float64Array, options: DspScanOptions = {}): Float64Array {
    const driver = resolveDriver(options.backend ?? this.backend);
    const out = options.out ?? new Float64Array(input.length);
    driver.mufdnSlice(
      input,
      out,
      this.delaySamples,
      this.gLin,
      this.sigmaKnee,
      this.vCeil,
      this.rDc,
    );
    return out;
  }

  public static scan(
    input: Float64Array,
    params: ModulatedUfdnParams = {},
    options: DspScanOptions = {},
  ): Float64Array {
    const ufdn = new ModulatedUfdn(params, options.backend);
    return ufdn.process(input, options);
  }
}

/**
 * Evaluates the $3\text{ kHz}$ 4-FMA + Algebraic Newton unit-circle rotor (`[\cos\theta, \sin\theta]`)
 * for a scalar `number` or vectorized `Float64Array` (returning a $2N$-element `Float64Array`
 * with the `cos` row followed by the `sin` row).
 */
export function unitRotor(phi: number, backend?: BackendChoice): [number, number];
export function unitRotor(
  phi: Float64Array,
  out?: Float64Array,
  backend?: BackendChoice,
): Float64Array;
export function unitRotor(
  phi: number | Float64Array,
  outOrBackend?: Float64Array | BackendChoice,
  maybeBackend?: BackendChoice,
): [number, number] | Float64Array {
  if (typeof phi === "number") {
    const backend = typeof outOrBackend === "string" ? outOrBackend : undefined;
    const driver = resolveDriver(backend);
    return driver.unitRotor(phi);
  }
  const out =
    outOrBackend instanceof Float64Array ? outOrBackend : new Float64Array(phi.length * 2);
  const backend = typeof outOrBackend === "string" ? outOrBackend : maybeBackend;
  const driver = resolveDriver(backend);
  driver.unitRotorSlice(phi, out);
  return out;
}

/**
 * Evaluates the 4-tap Cubic Lagrange fractional delay weights $[w_0(\mu), w_1(\mu), w_2(\mu), w_3(\mu)]$ (`CL-06`).
 */
export function lagrangeWeights(
  mu: number,
  backend?: BackendChoice,
): [number, number, number, number] {
  const driver = resolveDriver(backend);
  return driver.cubicLagrangeWeights(mu);
}

/**
 * Applies the 8-channel Orthogonal Givens + Householder mixing matrix $\mathbf{U}(\psi)$ (`CL-08`).
 */
export function givensHouseholderMix(
  x: Float64Array | readonly number[],
  cosPsi = 1.0,
  sinPsi = 0.0,
  backend?: BackendChoice,
): Float64Array {
  const driver = resolveDriver(backend);
  const xF64 = x instanceof Float64Array ? x : new Float64Array(x);
  return driver.givensHouseholderMix8ch(xF64, cosPsi, sinPsi);
}
