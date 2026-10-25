import {
  AccelerationBadge,
  Badge,
  Button,
  CapturePlanPanel,
  ClampAuditVisualizer,
  ControlBusVisualizer,
  DnamAudioHostProvider,
  DnamPluginRack,
  KroneckerResumeVisualizer,
  ModulatedUfdnVisualizer,
  PolyAdaaVisualizer,
  RgLruPoleVisualizer,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
  TrainingTimelinePanel,
  ZohRescalerVisualizer,
} from "@dnam-project/ui";
import { ExternalLink, FolderOpen, Layers } from "lucide-react";
import { TauriNativeAudioHost } from "../host.js";
import { DEFAULT_STUDIO_VIEW_MODEL, type StudioWorkspaceViewModel } from "./studio_store.js";

export interface StudioVisualizerState {
  readonly drive: number;
  readonly freqHz: number;
  readonly blend: number;
  readonly sampleIndex: number;
  readonly nodes: number;
  readonly lagError: number;
  readonly r2: number;
  readonly sampleRateHz: number;
}

export interface StudioVisualizerCallbacks {
  readonly onDriveChange?: ((drive: number) => void) | undefined;
  readonly onFreqChange?: ((freqHz: number) => void) | undefined;
  readonly onBlendChange?: ((blend: number) => void) | undefined;
  readonly onSampleIndexChange?: ((sampleIndex: number) => void) | undefined;
  readonly onNodesChange?: ((nodes: number) => void) | undefined;
  readonly onLagErrorChange?: ((lagError: number) => void) | undefined;
  readonly onR2Change?: ((r2: number) => void) | undefined;
  readonly onSampleRateChange?: ((sampleRateHz: number) => void) | undefined;
}

export interface StudioDesktopActions {
  readonly onOpenModelDialog?: (() => void) | undefined;
  readonly onRevealExport?: (() => void) | undefined;
  readonly onOpenReport?: (() => void) | undefined;
}

export interface StudioAppProps {
  readonly viewModel?: StudioWorkspaceViewModel | undefined;
  readonly host?: TauriNativeAudioHost | undefined;
  readonly defaultTab?: "rack" | "capture" | "trainer" | "visualizers" | undefined;
  readonly visualizerState?: StudioVisualizerState | undefined;
  readonly visualizerCallbacks?: StudioVisualizerCallbacks | undefined;
  readonly desktopActions?: StudioDesktopActions | undefined;
  readonly desktopStatusMessage?: string | undefined;
  readonly onCaptureStep?: ((stemId: string) => void) | undefined;
  readonly onStartTraining?: (() => void) | undefined;
}

export function StudioApp({
  viewModel = DEFAULT_STUDIO_VIEW_MODEL,
  host,
  defaultTab = "rack",
  visualizerState,
  visualizerCallbacks,
  desktopActions,
  desktopStatusMessage,
  onCaptureStep,
  onStartTraining,
}: StudioAppProps) {
  const resolvedHost = host ?? new TauriNativeAudioHost();

  return (
    <DnamAudioHostProvider host={resolvedHost}>
      <div
        data-testid="dnam-studio-app"
        className="min-h-screen bg-slate-950 p-6 font-sans text-slate-100"
      >
        <header className="mb-6 flex flex-col gap-3 border-b border-slate-800 pb-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2.5">
            <Layers className="size-5 text-cyan-400" />
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-base font-semibold tracking-tight">
                  {viewModel.workspaceName}
                </h1>
                <Badge variant="outline">dnam:// Single-Instance</Badge>
              </div>
              <p className="text-xs text-slate-400">
                {desktopStatusMessage ??
                  "Tauri v2 Native Desktop Studio + Headless dnam-train Orchestrator"}
              </p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              size="sm"
              variant="outline"
              data-testid="studio-open-dialog-btn"
              onClick={desktopActions?.onOpenModelDialog}
            >
              <FolderOpen className="size-3.5" />
              Open .dnam
            </Button>
            <Button
              size="sm"
              variant="outline"
              data-testid="studio-reveal-export-btn"
              onClick={desktopActions?.onRevealExport}
            >
              <ExternalLink className="size-3.5" />
              Reveal Export
            </Button>
            <Button
              size="sm"
              variant="secondary"
              data-testid="studio-open-report-btn"
              onClick={desktopActions?.onOpenReport}
            >
              <ExternalLink className="size-3.5" />
              Open Report
            </Button>
            <AccelerationBadge backend={resolvedHost.backend} />
          </div>
        </header>

        <h2 className="sr-only">Studio Workspace</h2>
        <Tabs defaultValue={defaultTab}>
          <TabsList aria-label="Studio Workspace Tabs">
            <TabsTrigger value="rack">Rack Audition</TabsTrigger>
            <TabsTrigger value="capture">Capture Schedule</TabsTrigger>
            <TabsTrigger value="trainer">Python Trainer (dnam-train)</TabsTrigger>
            <TabsTrigger value="visualizers">Tri-Physics Visualizers</TabsTrigger>
          </TabsList>

          <TabsContent value="rack">
            <DnamPluginRack model={viewModel.rackModel} />
          </TabsContent>

          <TabsContent value="capture">
            <CapturePlanPanel
              modelId={viewModel.rackModel.modelId}
              steps={viewModel.captureSteps}
              onCaptureStep={onCaptureStep}
            />
          </TabsContent>

          <TabsContent value="trainer">
            <TrainingTimelinePanel
              timeline={viewModel.trainingTimeline}
              onStartTraining={onStartTraining}
            />
          </TabsContent>

          <TabsContent value="visualizers">
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <PolyAdaaVisualizer
                drive={visualizerState?.drive}
                freqHz={visualizerState?.freqHz}
                onDriveChange={visualizerCallbacks?.onDriveChange}
                onFreqChange={visualizerCallbacks?.onFreqChange}
              />
              <RgLruPoleVisualizer
                blend={visualizerState?.blend}
                onBlendChange={visualizerCallbacks?.onBlendChange}
              />
              <ModulatedUfdnVisualizer
                sampleIndex={visualizerState?.sampleIndex}
                onSampleIndexChange={visualizerCallbacks?.onSampleIndexChange}
              />
              <ControlBusVisualizer
                nodes={visualizerState?.nodes}
                onNodesChange={visualizerCallbacks?.onNodesChange}
              />
              <KroneckerResumeVisualizer
                lagError={visualizerState?.lagError}
                onLagErrorChange={visualizerCallbacks?.onLagErrorChange}
              />
              <ClampAuditVisualizer
                r2={visualizerState?.r2}
                onR2Change={visualizerCallbacks?.onR2Change}
              />
              <ZohRescalerVisualizer
                sampleRateHz={visualizerState?.sampleRateHz}
                onSampleRateChange={visualizerCallbacks?.onSampleRateChange}
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </DnamAudioHostProvider>
  );
}
