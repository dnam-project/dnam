import { beforeAll, describe, expect, it } from "vitest";
import { ensureWasmReady, getWasmBytes } from "../../tests/helpers.js";
import {
  type AudioWorkletContextLike,
  type AudioWorkletNodeLike,
  createDnamSharedBuffer,
  DNAM_SAB_BYTE_LENGTH,
  DnamAudioWorkletProcessor,
  DnamContainer,
  type DnamSharedBridge,
  DnamSharedBridge as DnamSharedBridgeClass,
  DnamWorkletController,
  type DnamWorkletProcessorOptions,
  isSharedArrayBufferSupported,
  type WorkletInboundMessage,
  type WorkletOutboundMessage,
} from "../index.js";

interface RefControls {
  gain: number;
  boost: "off" | "on" | "ultra";
  [key: string]: number | string;
}

class LinkedWorkletHarness implements AudioWorkletNodeLike {
  public readonly processor: DnamAudioWorkletProcessor;
  public readonly port: {
    onmessage: ((event: { data: WorkletOutboundMessage }) => void) | null;
    postMessage(message: WorkletInboundMessage): void;
  };
  public disconnected = false;

  public constructor(
    _context: AudioWorkletContextLike,
    _name: string,
    options?: {
      processorOptions?: DnamWorkletProcessorOptions;
    },
  ) {
    this.port = {
      onmessage: null,
      postMessage: (msg: WorkletInboundMessage) => {
        this.processor.port.onmessage?.({ data: msg });
      },
    };
    this.processor = new DnamAudioWorkletProcessor(options);
    this.processor.port.postMessage = (msg: WorkletOutboundMessage) => {
      this.port.onmessage?.({ data: msg });
    };
  }

  public disconnect(): void {
    this.disconnected = true;
    this.processor.dispose();
  }
}

describe("WebAudio AudioWorklet Processor, SharedArrayBuffer Bridge & Controller", () => {
  beforeAll(async () => {
    await ensureWasmReady();
  });

  it("reads and writes lock-free 256-byte SharedArrayBuffer and ArrayBuffer fallback", () => {
    expect(isSharedArrayBufferSupported()).toBe(true);
    expect(() => new DnamSharedBridgeClass(new ArrayBuffer(64))).toThrow(/at least 256 bytes/);

    const sab = createDnamSharedBuffer(1, [5.0, 1.0]);
    const sabBridge: DnamSharedBridge = new DnamSharedBridgeClass(sab);
    expect(sabBridge.getTargetTier()).toBe(1);
    expect(sabBridge.readTargetControl(0)).toBeCloseTo(5.0, 5);
    expect(sabBridge.readTargetControl(1)).toBeCloseTo(1.0, 5);
    expect(sabBridge.consumeImmediateMask()).toBe(0b11);
    expect(sabBridge.consumeImmediateMask()).toBe(0);

    // Non-Atomics ArrayBuffer fallback path
    const ab = new ArrayBuffer(DNAM_SAB_BYTE_LENGTH);
    const abBridge = new DnamSharedBridgeClass(ab);
    abBridge.writeTier(2);
    abBridge.writeControl(0, 7.5, false);
    abBridge.writeControl(1, 2.0, true);
    abBridge.writeControl(99, 1.0, true); // Out-of-bounds no-op
    abBridge.triggerReset();
    expect(abBridge.getTargetTier()).toBe(2);
    expect(abBridge.getResetSeq()).toBe(1);
    expect(abBridge.readTargetControl(0)).toBeCloseTo(7.5, 5);
    expect(abBridge.consumeImmediateMask()).toBe(0b10);

    abBridge.publishTelemetry({
      activeTier: 2,
      activeClampMask: 3,
      controlDecimationHost: 16,
      maxAdaaDrive: 0.4,
      maxPoleRadius: 0.92,
      maxEffectiveLoopGain: 0.85,
      tailTimeMs: 450,
      zHead: [0.1, 0.2, 0.3, 0.4],
      envStates: [0.01, 0.02, 0.03, 0.04],
      lfoHarmonics: [1, 0, 1, 0, 1, 0, 1, 0],
      clampedControls: Array.from({ length: 16 }, (_, i) => i * 0.5),
    });
    const telem = abBridge.readTelemetry();
    expect(telem.activeTier).toBe(2);
    expect(telem.activeClampMask).toBe(3);
    expect(telem.tailTimeMs).toBeCloseTo(450, 3);
    expect(telem.zHead[0]).toBeCloseTo(0.1, 5);
    expect(telem.clampedControls[2]).toBeCloseTo(1.0, 5);
  });

  it("synchronizes DnamWorkletController and DnamAudioWorkletProcessor via Lock-Free SharedArrayBuffer", async () => {
    const container = DnamContainer.createReference(true, "wasm");
    let moduleAdded = false;
    const mockContext: AudioWorkletContextLike = {
      sampleRate: 48_000,
      audioWorklet: {
        addModule: async () => {
          moduleAdded = true;
        },
      },
    };

    const controller = await DnamWorkletController.createNode<RefControls>(mockContext, container, {
      tier: "full",
      wasmSource: getWasmBytes(),
      workletModuleUrl: "/worklet-processor.js",
      useSharedArrayBuffer: true,
      telemetryIntervalBlocks: 2,
      NodeConstructor: LinkedWorkletHarness,
    });

    expect(moduleAdded).toBe(true);
    expect(controller.usesSharedMemory).toBe(true);
    expect(controller.activeTier).toBe(1);
    expect(controller.activeTierName).toBe("full");
    expect(controller.container).toBe(container);

    const harness = controller.node as LinkedWorkletHarness;
    const inL = new Float32Array(128).fill(0.1);
    const inR = new Float32Array(128).fill(-0.1);
    const outL = new Float32Array(128);
    const outR = new Float32Array(128);

    // Update controls & tier via SharedArrayBuffer
    controller.setControls({ gain: 7.0, boost: "on" }, { immediate: false });
    controller.setTier("ultra");
    controller.reset();

    // Run 2 blocks of 128 samples so telemetry publishes back to SharedArrayBuffer
    expect(harness.processor.process([[inL, inR]], [[outL, outR]])).toBe(true);
    expect(harness.processor.process([[inL]], [[outL, outR]])).toBe(true);
    expect(harness.processor.process([[inL]], [[outL]])).toBe(true);
    expect(harness.processor.process([], [])).toBe(true);

    expect(controller.activeTier).toBe(2);
    expect(controller.activeTierName).toBe("ultra");
    const telem = controller.telemetry();
    expect(telem.activeTier).toBe(2);

    // Hot-swap reload in SharedArrayBuffer mode
    const cLite = DnamContainer.createReference(false, "wasm");
    controller.reload(cLite.bytes, { tier: "lite", sampleRateHz: 48_000 });
    expect(harness.processor.process([[inL, inR]], [[outL, outR]])).toBe(true);
    expect(harness.processor.process([[inL, inR]], [[outL, outR]])).toBe(true);
    expect(controller.activeTierName).toBe("lite");

    controller[Symbol.dispose]();
    expect(harness.disconnected).toBe(true);
  });

  it("synchronizes DnamWorkletController and DnamAudioWorkletProcessor via MessagePort fallback", async () => {
    const container = DnamContainer.createReference(true, "wasm");
    const mockContext: AudioWorkletContextLike = {
      sampleRate: 48_000,
    };

    const controller = await DnamWorkletController.createNode<RefControls>(
      mockContext,
      container.bytes,
      {
        tier: "full",
        wasmSource: getWasmBytes(),
        useSharedArrayBuffer: false,
        NodeConstructor: LinkedWorkletHarness,
      },
    );

    expect(controller.usesSharedMemory).toBe(false);
    expect(controller.activeTierName).toBe("full");

    const harness = controller.node as LinkedWorkletHarness;
    controller.setControls({ gain: 6.5, boost: "on" }, { immediate: true });
    controller.setControls({ gain: 5.5 }, { immediate: false });
    controller.setTier("lite");
    controller.reset();

    // Also exercise batch setControls message and error message handling on processor
    harness.port.postMessage({
      type: "setControls",
      entries: [
        { index: 0, value: 6.0 },
        { index: 1, value: 1.0 },
      ],
      immediate: true,
    });
    harness.port.postMessage({
      type: "setControls",
      entries: [{ index: 0, value: 5.0 }],
      immediate: false,
    });

    const telem = controller.telemetry();
    expect(telem.activeTier).toBe(0);
    expect(controller.activeTierName).toBe("lite");

    controller.reload(container, { tier: "ultra", sampleRateHz: 48_000 });
    expect(controller.activeTierName).toBe("ultra");

    // Verify invalid reload triggers error outbound message without crashing processor
    let receivedError = false;
    const origOnMessage = harness.port.onmessage;
    harness.port.onmessage = (ev) => {
      if (ev.data.type === "error") {
        receivedError = true;
      }
      origOnMessage?.(ev);
    };
    harness.port.postMessage({
      type: "reload",
      containerBytes: new Uint8Array(16),
    });
    expect(receivedError).toBe(true);

    controller.dispose();
  });
});
