import {
  createContext,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type { DnamAudioHost, DnamTelemetrySnapshot } from "./types.js";
import { WebAudioWorkletHost } from "./web_worklet_host.js";

const DnamAudioHostContext = createContext<DnamAudioHost | null>(null);

export interface DnamAudioHostProviderProps {
  readonly host?: DnamAudioHost | undefined;
  readonly children: ReactNode;
}

export function DnamAudioHostProvider({ host, children }: DnamAudioHostProviderProps) {
  const resolvedHost = useMemo<DnamAudioHost>(() => host ?? new WebAudioWorkletHost(), [host]);
  return (
    <DnamAudioHostContext.Provider value={resolvedHost}>{children}</DnamAudioHostContext.Provider>
  );
}

export interface UseDnamAudioHostResult {
  readonly host: DnamAudioHost;
  readonly telemetry: DnamTelemetrySnapshot;
  readonly setParameter: (id: string, value: number) => void;
  readonly setSwitchIndex: (id: string, index: number) => void;
}

export function useDnamAudioHost(): UseDnamAudioHostResult {
  const ctxHost = useContext(DnamAudioHostContext);
  const fallbackHost = useMemo(() => new WebAudioWorkletHost(), []);
  const host = ctxHost ?? fallbackHost;
  const [telemetry, setTelemetry] = useState<DnamTelemetrySnapshot>(() => host.getTelemetry());

  useEffect(() => {
    setTelemetry(host.getTelemetry());
    return host.subscribe((snapshot) => {
      setTelemetry(snapshot);
    });
  }, [host]);

  const setParameter = useCallback(
    (id: string, value: number) => {
      host.setParameter(id, value);
    },
    [host],
  );

  const setSwitchIndex = useCallback(
    (id: string, index: number) => {
      host.setSwitchIndex(id, index);
    },
    [host],
  );

  return {
    host,
    telemetry,
    setParameter,
    setSwitchIndex,
  };
}
