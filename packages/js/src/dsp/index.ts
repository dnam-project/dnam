/**
 * Curated exports for `@dnam-project/dnam/dsp` (`ref[sdk.ts.layout.curated-exports]`,
 * `ref[sdk.ts.dsp.reference-kernels]`).
 *
 * @module
 */

export {
  chebyshevLfo,
  chebyshevLfoSlice,
  lobattoBasis,
  lobattoBasisSlice,
  rank4Fm,
  rank4FmSlice,
  trustRegionSoftclip,
  trustRegionSoftclipSlice,
} from "./control-bus.js";
export {
  cubicLagrangeWeights,
  cubicLagrangeWeightsSlice,
  givensHouseholder8ch,
  givensHouseholder8chSlice,
  givensHouseholder16ch,
  givensHouseholder16chSlice,
} from "./mixing.js";
export {
  type ModalParams,
  modalSlice,
  modalStep,
  unitRotor,
  unitRotorSlice,
} from "./modal.js";
export {
  type PolyAdaaParams,
  polyAdaaAlgebraic,
  polyAdaaSlice,
  polyAdaaStep,
} from "./nonlinear.js";
export {
  catmullRomHermite,
  catmullRomHermiteSlice,
  coerciveStep,
  tankSlice,
} from "./transport.js";
