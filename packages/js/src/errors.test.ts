import { describe, expect, it } from "vitest";
import {
  DnamContainerError,
  DnamControlError,
  DnamError,
  DnamSampleRateError,
  DnamValidationError,
  mapTaggedError,
  rethrowDnamError,
} from "./errors.js";

describe("DnamError hierarchy and tagged FFI error mapping", () => {
  it("maps tagged Rust/FFI errors to exact DnamError subclasses", () => {
    const cErr = mapTaggedError("[DNAM-CONTAINER] Bad magic");
    expect(cErr).toBeInstanceOf(DnamContainerError);
    expect(cErr.code).toBe("DNAM_CONTAINER_ERROR");
    expect(cErr.message).toBe("Bad magic");

    const sErr = mapTaggedError("[DNAM-SAMPLERATE] Unsupported 12000 Hz");
    expect(sErr).toBeInstanceOf(DnamSampleRateError);

    const ctrlErr = mapTaggedError("[DNAM-CONTROL] Invalid control");
    expect(ctrlErr).toBeInstanceOf(DnamControlError);

    const valErr = mapTaggedError("[DNAM-VALIDATION] Unaligned offset");
    expect(valErr).toBeInstanceOf(DnamValidationError);

    const genErr = mapTaggedError("Untagged error");
    expect(genErr).toBeInstanceOf(DnamError);

    expect(() => rethrowDnamError(cErr)).toThrow(DnamContainerError);
    expect(() => rethrowDnamError(new Error("[DNAM-SAMPLERATE] 8000"))).toThrow(
      DnamSampleRateError,
    );
    expect(() => rethrowDnamError("[DNAM-CONTROL] bad")).toThrow(DnamControlError);
  });
});
