import {
  ClampAuditVisualizer,
  ControlBusVisualizer,
  KroneckerResumeVisualizer,
  ModulatedUfdnVisualizer,
  PolyAdaaVisualizer,
  RgLruPoleVisualizer,
  ZohRescalerVisualizer,
} from "./index.js";

export default {
  "1. Core 1: 1x B-Spline PolyADAA": <PolyAdaaVisualizer />,
  "2. Core 1: Radial-Polar RG-LRU Pole Trajectory": <RgLruPoleVisualizer />,
  "3. Core 2: Modulated UFDN Dual-Tap Crossfade": <ModulatedUfdnVisualizer />,
  "4. Control Bus: Lobatto HDMR-FM & 3 kHz LFO": <ControlBusVisualizer />,
  "5. Iterative Resume: Surrogate Delay & Kronecker": <KroneckerResumeVisualizer />,
  "6. Stability Registry: Mahalanobis Trust Region": <ClampAuditVisualizer />,
  "7. Container: Continuous-Time ZOH Rescaler": <ZohRescalerVisualizer />,
};
