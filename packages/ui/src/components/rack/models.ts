import type { TriPhysicsDomain } from "../../tokens.js";
import type { DiscreteSwitchOption } from "../controls/DiscreteSwitch.js";

export interface RackParameterDescriptor {
  readonly id: string;
  readonly label: string;
  readonly min: number;
  readonly max: number;
  readonly step?: number | undefined;
  readonly defaultValue: number;
  readonly observedMin?: number | undefined;
  readonly observedMax?: number | undefined;
  readonly unit?: string | undefined;
  readonly domain: TriPhysicsDomain;
}

export interface RackSwitchDescriptor {
  readonly id: string;
  readonly label: string;
  readonly options: readonly DiscreteSwitchOption[];
  readonly defaultIndex: number;
}

export interface DnamPluginRackModel {
  readonly modelId: string;
  readonly title: string;
  readonly subtitle: string;
  readonly parameters: readonly RackParameterDescriptor[];
  readonly switches: readonly RackSwitchDescriptor[];
}
