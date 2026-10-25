import { useFixtureInput, useFixtureSelect } from "react-cosmos/client";
import {
  Badge,
  Button,
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
  Progress,
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "./index.js";

function ButtonAndBadgeShowcase() {
  const [variant] = useFixtureSelect("Button Variant", {
    options: ["default", "secondary", "outline", "ghost", "destructive"],
    defaultValue: "default",
  });
  const [size] = useFixtureSelect("Button Size", {
    options: ["default", "sm", "lg", "icon"],
    defaultValue: "default",
  });
  const [label] = useFixtureInput("Button Label", "Engage Core 1");

  return (
    <Card className="max-w-xl">
      <CardHeader>
        <CardTitle>DNAM Base UI Primitives</CardTitle>
        <CardDescription>shadcn/ui on @base-ui/react + Tailwind CSS v4</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button variant={variant} size={size}>
            {label}
          </Button>
          <Badge variant="core1">Core 1</Badge>
          <Badge variant="core2">Core 2</Badge>
          <Badge variant="control">Control Bus</Badge>
          <Badge variant="safe">CL-01..CL-20 Safe</Badge>
          <Badge variant="warn">Trust Region</Badge>
          <Badge variant="danger">Clip Guard</Badge>
        </div>
      </CardContent>
      <CardFooter className="text-xs text-slate-400">
        Zero-runtime CSS token variables via @tailwindcss/vite
      </CardFooter>
    </Card>
  );
}

function ProgressAndTabsShowcase() {
  const [progress] = useFixtureInput("Progress (%)", 72);
  return (
    <Card className="max-w-xl">
      <CardHeader>
        <CardTitle>Progress & Workspace Tabs</CardTitle>
        <CardDescription>Accessible Base UI Progress & Tabs primitives</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <Progress aria-label="Model build progress" value={progress} />
        <Tabs defaultValue="core1">
          <TabsList aria-label="Core selector">
            <TabsTrigger value="core1">Core 1</TabsTrigger>
            <TabsTrigger value="core2">Core 2</TabsTrigger>
            <TabsTrigger value="control">Control Bus</TabsTrigger>
          </TabsList>
          <TabsContent value="core1">12-Layer PolyADAA + Radial-Polar RG-LRU</TabsContent>
          <TabsContent value="core2">Modulated Unitary Feedback Delay Network</TabsContent>
          <TabsContent value="control">3 kHz Lobatto HDMR + Rank-4 FM</TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}

export default {
  "Buttons & Domain Badges": <ButtonAndBadgeShowcase />,
  "Progress & Tabs": <ProgressAndTabsShowcase />,
};
