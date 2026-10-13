"use client";

import * as React from "react";
import {
  Zap,
  Search,
  Moon,
  Loader2,
  Brain,
  Sparkles,
  Layers,
  CheckCircle2,
  Clock,
  Cpu,
  ArrowRight,
  ShieldCheck,
  RefreshCw,
  Sliders,
} from "lucide-react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";

interface EncodeResult {
  engramId: string;
  status: string;
  type: string;
  salience: number;
  neuromodulation: {
    dopamine: string;
    acetylcholine: string;
  };
  vectorDims: number;
  decayHalfLife: string;
  timestamp: string;
}

interface RecallItem {
  id: string;
  query: string;
  content: string;
  relevance: number;
  salience: number;
  type: "episodic" | "semantic" | "procedural";
  fidelity: {
    label: string;
    score: number;
  };
  agent_id: string;
}

interface SleepReport {
  status: string;
  cycleId: string;
  swsReplayCount: number;
  transferredCount: number;
  prunedCount: number;
  plasticityDelta: string;
  compressionRatio: string;
  timestamp: string;
}

export default function PlaygroundPage() {
  // --- ENCODE TAB STATE ---
  const [encodeContent, setEncodeContent] = React.useState(
    "User requested integrating OriginAI Brain with autonomous LLM agent loop for persistent episodic learning across multi-turn sessions."
  );
  const [encodeAgentId, setEncodeAgentId] = React.useState("agent_alpha");
  const [encodeUserId, setEncodeUserId] = React.useState("user_megha");
  const [isEncoding, setIsEncoding] = React.useState(false);
  const [encodeResult, setEncodeResult] = React.useState<EncodeResult | null>(
    null
  );

  const handleEncode = () => {
    if (!encodeContent.trim()) return;
    setIsEncoding(true);
    setTimeout(() => {
      setEncodeResult({
        engramId: `engram_${Math.random().toString(36).substring(2, 9)}`,
        status: "Consolidated into Hippocampus",
        type: "Episodic",
        salience: 0.94,
        neuromodulation: {
          dopamine: "+0.32 (High novelty detected)",
          acetylcholine: "+0.45 (Active encoding state)",
        },
        vectorDims: 1536,
        decayHalfLife: "72 hours (standard pre-sleep decay curve)",
        timestamp: new Date().toLocaleTimeString(),
      });
      setIsEncoding(false);
    }, 800);
  };

  // --- RECALL TAB STATE ---
  const [recallQuery, setRecallQuery] = React.useState(
    "How does the dual-store architecture consolidate memory?"
  );
  const [isRecalling, setIsRecalling] = React.useState(false);
  const [recallResults, setRecallResults] = React.useState<RecallItem[] | null>(
    null
  );

  const handleRecall = () => {
    if (!recallQuery.trim()) return;
    setIsRecalling(true);
    setTimeout(() => {
      setRecallResults([
        {
          id: "rec-01",
          query: recallQuery,
          content:
            "OriginAI Brain integrates biologically inspired Dual-Store architecture: Hippocampus fast-learning index and Neocortex long-term consolidated store.",
          relevance: 0.964,
          salience: 0.95,
          type: "semantic",
          fidelity: { label: "High Fidelity", score: 98 },
          agent_id: "system_core",
        },
        {
          id: "rec-02",
          query: recallQuery,
          content:
            "Synaptic plasticity decay follows Ebbinghaus power law modified with neuromodulatory dopamine and acetylcholine scaling factors.",
          relevance: 0.882,
          salience: 0.89,
          type: "semantic",
          fidelity: { label: "Reconstructed", score: 87 },
          agent_id: "system_core",
        },
        {
          id: "rec-03",
          query: recallQuery,
          content:
            "Sleep consolidation protocol: 1. Sample high-salience episodic traces. 2. Replay patterns with noise injection. 3. Backprop into neocortex weights. 4. Prune decayed synapses.",
          relevance: 0.815,
          salience: 0.84,
          type: "procedural",
          fidelity: { label: "Pattern Completed", score: 82 },
          agent_id: "worker_daemon",
        },
      ]);
      setIsRecalling(false);
    }, 600);
  };

  // --- SLEEP TAB STATE ---
  const [isSleeping, setIsSleeping] = React.useState(false);
  const [sleepReport, setSleepReport] = React.useState<SleepReport | null>(
    null
  );

  const handleRunSleep = () => {
    setIsSleeping(true);
    setTimeout(() => {
      setSleepReport({
        status: "Consolidation Phase Completed",
        cycleId: `slp_${Math.random().toString(36).substring(2, 8)}`,
        swsReplayCount: 142,
        transferredCount: 38,
        prunedCount: 19,
        plasticityDelta: "+18.4% SNR enhancement",
        compressionRatio: "3.2x neocortical footprint reduction",
        timestamp: new Date().toLocaleTimeString(),
      });
      setIsSleeping(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">
          Brain Playground
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Simulate biological memory mechanics: encode new engrams, perform
          contextual recall with associative reconstruction, or trigger slow-wave
          sleep consolidation cycles.
        </p>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="encode" className="w-full space-y-6">
        <TabsList className="grid grid-cols-3 w-full max-w-md bg-muted/60 p-1 border border-border">
          <TabsTrigger
            value="encode"
            className="flex items-center justify-center gap-2 font-medium"
          >
            <Zap className="size-4 text-amber-400" />
            <span>Encode</span>
          </TabsTrigger>
          <TabsTrigger
            value="recall"
            className="flex items-center justify-center gap-2 font-medium"
          >
            <Search className="size-4 text-cyan-400" />
            <span>Recall</span>
          </TabsTrigger>
          <TabsTrigger
            value="sleep"
            className="flex items-center justify-center gap-2 font-medium"
          >
            <Moon className="size-4 text-indigo-400" />
            <span>Sleep</span>
          </TabsTrigger>
        </TabsList>

        {/* ----------------- ENCODE TAB ----------------- */}
        <TabsContent value="encode" className="space-y-6 outline-none">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Input Card */}
            <Card className="lg:col-span-7 border-border bg-card/70">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Zap className="size-4 text-amber-400" />
                  Encode New Memory Engram
                </CardTitle>
                <CardDescription>
                  Inputs are parsed into cognitive engrams, tagged with emotional
                  and neuromodulatory salience, and written to hippocampal working
                  memory.
                </CardDescription>
              </CardHeader>

              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs font-medium text-foreground">
                    Memory Content
                  </label>
                  <Textarea
                    placeholder="Enter observation, experience, or factual premise..."
                    className="min-h-32 bg-background/50 text-sm font-sans"
                    value={encodeContent}
                    onChange={(e) => setEncodeContent(e.target.value)}
                  />
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-foreground">
                      Agent ID
                    </label>
                    <Input
                      placeholder="e.g. agent_alpha"
                      value={encodeAgentId}
                      onChange={(e) => setEncodeAgentId(e.target.value)}
                      className="bg-background/50 h-9 font-mono text-xs"
                    />
                  </div>
                  <div className="space-y-1.5">
                    <label className="text-xs font-medium text-foreground">
                      User ID
                    </label>
                    <Input
                      placeholder="e.g. user_megha"
                      value={encodeUserId}
                      onChange={(e) => setEncodeUserId(e.target.value)}
                      className="bg-background/50 h-9 font-mono text-xs"
                    />
                  </div>
                </div>
              </CardContent>

              <CardFooter className="flex justify-between items-center border-t border-border/50 pt-4">
                <span className="text-xs text-muted-foreground font-mono">
                  Target: Hippocampus (Fast Index)
                </span>
                <Button
                  onClick={handleEncode}
                  disabled={isEncoding || !encodeContent.trim()}
                  className="gap-2 font-medium"
                >
                  {isEncoding ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Encoding...
                    </>
                  ) : (
                    <>
                      <Zap className="size-4 text-amber-300" />
                      Encode Memory
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>

            {/* Result Panel */}
            <Card className="lg:col-span-5 border-border bg-card/70 flex flex-col justify-between">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <Sparkles className="size-4 text-amber-400" />
                    Encoding Result
                  </CardTitle>
                  {encodeResult && (
                    <Badge
                      variant="secondary"
                      className="bg-emerald-500/10 text-emerald-400 border-emerald-500/20 text-xs"
                    >
                      Success
                    </Badge>
                  )}
                </div>
                <CardDescription>
                  Real-time synaptic metrics computed upon ingestion.
                </CardDescription>
              </CardHeader>

              <CardContent className="flex-1 flex flex-col justify-center">
                {isEncoding ? (
                  <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
                    <Loader2 className="size-8 animate-spin text-amber-400" />
                    <p className="text-sm font-medium text-foreground">
                      Synthesizing neural embeddings...
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Evaluating dopamine novelty metrics & vector coordinates
                    </p>
                  </div>
                ) : encodeResult ? (
                  <div className="space-y-3.5 text-xs">
                    <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Engram ID:</span>
                        <span className="font-mono text-zinc-200 font-medium">
                          {encodeResult.engramId}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Status:</span>
                        <span className="text-emerald-400 font-medium">
                          {encodeResult.status}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Memory Type:</span>
                        <Badge
                          variant="secondary"
                          className="bg-indigo-500/10 text-indigo-400 text-[10px]"
                        >
                          {encodeResult.type}
                        </Badge>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">
                          Calculated Salience:
                        </span>
                        <span className="font-mono text-amber-400 font-semibold">
                          {encodeResult.salience}
                        </span>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800 space-y-2">
                      <div className="text-[11px] font-medium text-zinc-300">
                        Neuromodulatory Tone
                      </div>
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span>Dopamine (DA):</span>
                        <span className="font-mono text-zinc-300">
                          {encodeResult.neuromodulation.dopamine}
                        </span>
                      </div>
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span>Acetylcholine (ACh):</span>
                        <span className="font-mono text-zinc-300">
                          {encodeResult.neuromodulation.acetylcholine}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-muted-foreground text-[11px]">
                      <div className="p-2 rounded bg-zinc-900/40 border border-zinc-800/80">
                        Dimensions:{" "}
                        <span className="font-mono text-zinc-200">
                          {encodeResult.vectorDims}
                        </span>
                      </div>
                      <div className="p-2 rounded bg-zinc-900/40 border border-zinc-800/80">
                        Half-life:{" "}
                        <span className="font-mono text-zinc-200">72 hrs</span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-10 text-center gap-2 border border-dashed border-border/80 rounded-lg">
                    <Brain className="size-8 text-muted-foreground/40" />
                    <p className="text-xs text-muted-foreground">
                      Click &ldquo;Encode Memory&rdquo; to simulate engram storage.
                    </p>
                  </div>
                )}
              </CardContent>

              {encodeResult && (
                <CardFooter className="text-[11px] text-muted-foreground border-t border-border/40 py-2.5">
                  Processed at {encodeResult.timestamp}
                </CardFooter>
              )}
            </Card>
          </div>
        </TabsContent>

        {/* ----------------- RECALL TAB ----------------- */}
        <TabsContent value="recall" className="space-y-6 outline-none">
          {/* Query Bar */}
          <Card className="border-border bg-card/70">
            <CardHeader className="pb-3">
              <CardTitle className="text-base flex items-center gap-2">
                <Search className="size-4 text-cyan-400" />
                Query Associative Memory
              </CardTitle>
              <CardDescription>
                Triggers pattern completion and engram reconstruction across
                semantic schemas and episodic instances.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex flex-col sm:flex-row gap-3">
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
                  <Input
                    placeholder="Enter query prompt (e.g., dual-store architecture, decay laws)..."
                    value={recallQuery}
                    onChange={(e) => setRecallQuery(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") handleRecall();
                    }}
                    className="pl-9 bg-background/50 h-10"
                  />
                </div>
                <Button
                  onClick={handleRecall}
                  disabled={isRecalling || !recallQuery.trim()}
                  className="gap-2 h-10 px-5"
                >
                  {isRecalling ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Searching...
                    </>
                  ) : (
                    <>
                      <Search className="size-4" />
                      Recall Memories
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Results Area */}
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-foreground flex items-center gap-2">
                <span>Retrieved Engrams</span>
                {recallResults && (
                  <Badge variant="outline" className="text-xs font-mono">
                    {recallResults.length} matches
                  </Badge>
                )}
              </h3>
            </div>

            {isRecalling ? (
              <div className="flex flex-col items-center justify-center py-16 gap-3 text-center border border-dashed border-border/80 rounded-xl bg-card/30">
                <Loader2 className="size-8 animate-spin text-cyan-400" />
                <p className="text-sm font-medium text-foreground">
                  Traversing synaptic associative graph...
                </p>
                <p className="text-xs text-muted-foreground">
                  Executing hippocampal pattern completion & neocortical alignment
                </p>
              </div>
            ) : recallResults ? (
              <div className="grid grid-cols-1 gap-4">
                {recallResults.map((item, index) => (
                  <Card
                    key={item.id}
                    className="border-border bg-card/70 hover:bg-card hover:border-zinc-700 transition-all"
                  >
                    <CardHeader className="pb-2">
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-zinc-800 text-zinc-300">
                            #{index + 1}
                          </span>
                          <Badge
                            variant="secondary"
                            className={
                              item.type === "semantic"
                                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 capitalize font-mono text-xs"
                                : "bg-amber-500/10 text-amber-400 border border-amber-500/20 capitalize font-mono text-xs"
                            }
                          >
                            {item.type}
                          </Badge>
                          <Badge
                            variant="outline"
                            className="text-xs font-mono border-cyan-500/30 text-cyan-400 bg-cyan-500/5"
                          >
                            {item.fidelity.label} ({item.fidelity.score}%)
                          </Badge>
                        </div>

                        <div className="flex items-center gap-3 text-xs font-mono">
                          <span className="text-muted-foreground">
                            agent:{" "}
                            <span className="text-zinc-300 font-medium">
                              {item.agent_id}
                            </span>
                          </span>
                        </div>
                      </div>
                    </CardHeader>

                    <CardContent className="space-y-3 pt-1">
                      <p className="text-sm text-foreground/90 leading-relaxed font-sans">
                        {item.content}
                      </p>

                      <div className="flex flex-wrap items-center gap-4 pt-3 border-t border-border/60 text-xs">
                        <div className="flex items-center gap-1.5">
                          <span className="text-muted-foreground">
                            Cosine Relevance:
                          </span>
                          <span className="font-mono font-semibold text-cyan-400">
                            {(item.relevance * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-muted-foreground">Salience:</span>
                          <span className="font-mono font-semibold text-amber-400">
                            {item.salience.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span className="text-muted-foreground">
                            Activation:
                          </span>
                          <span className="font-mono text-zinc-300">
                            Hippocampal Seed Node
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            ) : (
              <Card className="border-dashed border-border py-12 text-center bg-card/30">
                <CardContent className="flex flex-col items-center justify-center gap-2">
                  <Search className="size-8 text-muted-foreground/50" />
                  <p className="text-sm font-medium text-foreground">
                    No active recall query
                  </p>
                  <p className="text-xs text-muted-foreground">
                    Click &ldquo;Recall Memories&rdquo; to simulate memory
                    reconstruction.
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        </TabsContent>

        {/* ----------------- SLEEP TAB ----------------- */}
        <TabsContent value="sleep" className="space-y-6 outline-none">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Description & Action Card */}
            <Card className="lg:col-span-7 border-border bg-card/70 flex flex-col justify-between">
              <div>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Moon className="size-4 text-indigo-400" />
                    Slow-Wave Sleep Consolidation (SWS)
                  </CardTitle>
                  <CardDescription>
                    Biological brains don&apos;t just append logs — they sleep.
                    During SWS, episodic memory traces undergo neural replay,
                    semantic schema transfer, and synaptic downscaling.
                  </CardDescription>
                </CardHeader>

                <CardContent className="space-y-4 text-xs">
                  <div className="grid grid-cols-1 gap-3">
                    <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800 flex items-start gap-3">
                      <div className="p-1.5 rounded-md bg-indigo-500/10 text-indigo-400 shrink-0">
                        <RefreshCw className="size-4" />
                      </div>
                      <div className="space-y-1">
                        <h4 className="font-medium text-foreground text-xs">
                          1. SWS Sharp-Wave Replay
                        </h4>
                        <p className="text-muted-foreground leading-relaxed">
                          Hippocampal traces replay at 10x-20x biological speeds.
                          Noise is introduced to test invariance and generalization
                          across diverse semantic boundaries.
                        </p>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800 flex items-start gap-3">
                      <div className="p-1.5 rounded-md bg-emerald-500/10 text-emerald-400 shrink-0">
                        <Layers className="size-4" />
                      </div>
                      <div className="space-y-1">
                        <h4 className="font-medium text-foreground text-xs">
                          2. Episodic → Semantic Transfer
                        </h4>
                        <p className="text-muted-foreground leading-relaxed">
                          Frequent, high-salience episodes are distilled into
                          structured semantic world models and permanent neocortical
                          weights.
                        </p>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-zinc-900/60 border border-zinc-800 flex items-start gap-3">
                      <div className="p-1.5 rounded-md bg-rose-500/10 text-rose-400 shrink-0">
                        <Sliders className="size-4" />
                      </div>
                      <div className="space-y-1">
                        <h4 className="font-medium text-foreground text-xs">
                          3. Synaptic Pruning & Normalization
                        </h4>
                        <p className="text-muted-foreground leading-relaxed">
                          Low-salience ephemeral noise is decayed and pruned to
                          restore synaptic homeostasis and prevent catastrophic
                          interference.
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </div>

              <CardFooter className="border-t border-border/50 pt-4 flex items-center justify-between">
                <span className="text-xs text-muted-foreground font-mono">
                  Cycle Type: Full SWS Replay
                </span>
                <Button
                  onClick={handleRunSleep}
                  disabled={isSleeping}
                  className="gap-2 font-medium bg-indigo-600 hover:bg-indigo-500 text-white"
                >
                  {isSleeping ? (
                    <>
                      <Loader2 className="size-4 animate-spin" />
                      Consolidating (1.5s)...
                    </>
                  ) : (
                    <>
                      <Moon className="size-4" />
                      Run Sleep Cycle
                    </>
                  )}
                </Button>
              </CardFooter>
            </Card>

            {/* Sleep Report Panel */}
            <Card className="lg:col-span-5 border-border bg-card/70 flex flex-col justify-between">
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-base flex items-center gap-2">
                    <ShieldCheck className="size-4 text-indigo-400" />
                    Sleep Cycle Report
                  </CardTitle>
                  {sleepReport && (
                    <Badge
                      variant="secondary"
                      className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs"
                    >
                      Cycle Complete
                    </Badge>
                  )}
                </div>
                <CardDescription>
                  Summary metrics from hippocampal-neocortical consolidation.
                </CardDescription>
              </CardHeader>

              <CardContent className="flex-1 flex flex-col justify-center">
                {isSleeping ? (
                  <div className="flex flex-col items-center justify-center py-12 gap-3 text-center">
                    <Loader2 className="size-8 animate-spin text-indigo-400" />
                    <p className="text-sm font-medium text-foreground">
                      Running SWS Sharp-Wave Replay...
                    </p>
                    <p className="text-xs text-muted-foreground">
                      Transferring episodes to neocortex & applying synaptic
                      pruning
                    </p>
                  </div>
                ) : sleepReport ? (
                  <div className="space-y-3.5 text-xs">
                    <div className="p-3 rounded-lg bg-zinc-900/80 border border-zinc-800 space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Status:</span>
                        <span className="text-emerald-400 font-medium">
                          {sleepReport.status}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Cycle ID:</span>
                        <span className="font-mono text-zinc-200">
                          {sleepReport.cycleId}
                        </span>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 gap-2.5">
                      <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
                        <span className="text-muted-foreground">
                          SWS Replayed Engrams:
                        </span>
                        <span className="font-mono text-indigo-300 font-semibold text-sm">
                          {sleepReport.swsReplayCount} traces
                        </span>
                      </div>

                      <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
                        <span className="text-muted-foreground">
                          Episodic → Semantic Transfer:
                        </span>
                        <span className="font-mono text-emerald-300 font-semibold text-sm">
                          {sleepReport.transferredCount} schemas
                        </span>
                      </div>

                      <div className="p-3 rounded-lg bg-zinc-900/50 border border-zinc-800 flex items-center justify-between">
                        <span className="text-muted-foreground">
                          Synaptic Pruning & Cleaned:
                        </span>
                        <span className="font-mono text-rose-300 font-semibold text-sm">
                          {sleepReport.prunedCount} low-salience nodes
                        </span>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-zinc-900/30 border border-zinc-800/80 space-y-1 text-muted-foreground text-[11px]">
                      <div className="flex items-center justify-between">
                        <span>Plasticity Delta:</span>
                        <span className="text-zinc-200 font-mono">
                          {sleepReport.plasticityDelta}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span>Compression Ratio:</span>
                        <span className="text-zinc-200 font-mono">
                          {sleepReport.compressionRatio}
                        </span>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-10 text-center gap-2 border border-dashed border-border/80 rounded-lg">
                    <Moon className="size-8 text-muted-foreground/40" />
                    <p className="text-xs text-muted-foreground">
                      Click &ldquo;Run Sleep Cycle&rdquo; to execute SWS consolidation.
                    </p>
                  </div>
                )}
              </CardContent>

              {sleepReport && (
                <CardFooter className="text-[11px] text-muted-foreground border-t border-border/40 py-2.5">
                  Completed at {sleepReport.timestamp}
                </CardFooter>
              )}
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
