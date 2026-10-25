import "./styles.css";
import type { ReactNode } from "react";
import { DnamAudioHostProvider } from "./host/context.js";

export default function DnamCosmosDecorator({ children }: { readonly children: ReactNode }) {
  return (
    <DnamAudioHostProvider>
      <div className="min-h-screen bg-slate-950 p-6 font-sans text-slate-100 antialiased">
        {children}
      </div>
    </DnamAudioHostProvider>
  );
}
