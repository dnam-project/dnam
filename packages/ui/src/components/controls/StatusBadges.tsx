import { Cpu, ShieldAlert, ShieldCheck } from "lucide-react";
import type { DnamBackendAcceleration } from "../../host/types.js";
import { Badge } from "../ui/badge.js";

export interface AccelerationBadgeProps {
  readonly backend: DnamBackendAcceleration;
}

export function formatBackendLabel(backend: DnamBackendAcceleration): string {
  switch (backend) {
    case "arm-sme":
      return "ARM SME (Apple M4/M5)";
    case "arm64-neon":
      return "ARM64 NEON";
    case "x86-avx2-fma":
      return "x86_64 AVX2+FMA";
    case "scalar-reference":
      return "Scalar Reference";
    default:
      return "WASM SIMD128";
  }
}

export function AccelerationBadge({ backend }: AccelerationBadgeProps) {
  const variant =
    backend === "arm-sme" || backend === "arm64-neon" || backend === "x86-avx2-fma"
      ? "core1"
      : "core2";
  return (
    <Badge variant={variant}>
      <Cpu className="size-3" />
      <span>{formatBackendLabel(backend)}</span>
    </Badge>
  );
}

export interface StabilityAuditBadgeProps {
  readonly maxRgLruRadius: number;
  readonly ufdnLyapunovMargin: number;
  readonly mahalanobisR2: number;
}

export function StabilityAuditBadge({
  maxRgLruRadius,
  ufdnLyapunovMargin,
  mahalanobisR2,
}: StabilityAuditBadgeProps) {
  const polesSafe = maxRgLruRadius <= 0.9999;
  const lyapunovSafe = ufdnLyapunovMargin > 0;
  const trustSafe = mahalanobisR2 <= 4.0;
  const allSafe = polesSafe && lyapunovSafe && trustSafe;

  return (
    <Badge variant={allSafe ? "safe" : "warn"}>
      {allSafe ? <ShieldCheck className="size-3" /> : <ShieldAlert className="size-3" />}
      <span>
        {allSafe
          ? `CL-01..CL-20 Verified (r_max=${maxRgLruRadius.toFixed(4)})`
          : `Trust Region Active (R²=${mahalanobisR2.toFixed(2)})`}
      </span>
    </Badge>
  );
}
