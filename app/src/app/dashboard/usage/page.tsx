"use client";

import * as React from "react";
import {
  Zap,
  Search,
  Moon,
  Database,
  Cpu,
  ArrowUpRight,
  TrendingUp,
  CheckCircle2,
  Calendar,
  Sparkles,
  BarChart3,
  HardDrive,
} from "lucide-react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function UsagePage() {
  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 max-w-7xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">
          Usage & Quotas
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Monitor your brain operational consumption, API quota limits, and memory
          storage metrics for the current billing cycle.
        </p>
      </div>

      {/* Current Plan Card */}
      <Card className="border-border bg-gradient-to-r from-card/80 via-card/60 to-card/90 relative overflow-hidden">
        <div className="absolute right-0 top-0 translate-x-12 -translate-y-12 w-64 h-64 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

        <CardHeader className="pb-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="p-2.5 rounded-xl bg-primary/10 text-primary border border-primary/20">
                <Sparkles className="size-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs uppercase tracking-wider font-semibold text-muted-foreground">
                    Current Plan
                  </span>
                  <Badge
                    variant="secondary"
                    className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 text-xs font-mono font-medium"
                  >
                    Free
                  </Badge>
                </div>
                <CardTitle className="text-2xl font-bold mt-0.5">
                  Starter Plan
                </CardTitle>
              </div>
            </div>

            <Button className="gap-2 font-medium cursor-pointer shadow-sm">
              <ArrowUpRight className="size-4" />
              Upgrade to Pro
            </Button>
          </div>
        </CardHeader>

        <CardContent className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2 text-xs">
          <div className="flex items-center gap-2 text-muted-foreground">
            <CheckCircle2 className="size-4 text-emerald-400 shrink-0" />
            <span>1 Dedicated Brain Instance</span>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <CheckCircle2 className="size-4 text-emerald-400 shrink-0" />
            <span>Standard SWS Sleep Consolidation</span>
          </div>
          <div className="flex items-center gap-2 text-muted-foreground">
            <Calendar className="size-4 text-zinc-400 shrink-0" />
            <span>Quota resets in 18 days (Nov 1, 2026)</span>
          </div>
        </CardContent>
      </Card>

      {/* 2 Usage Meter Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Encode Operations Meter */}
        <Card className="border-border bg-card/70 flex flex-col justify-between">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-primary/10 text-primary">
                  <Zap className="size-4" />
                </div>
                <div>
                  <CardTitle className="text-base font-semibold">
                    Encode Operations
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Engram creation & sensory ingestion
                  </CardDescription>
                </div>
              </div>
              <Badge variant="outline" className="font-mono text-xs">
                24.5%
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="flex items-baseline justify-between">
              <div className="text-2xl font-bold font-mono tracking-tight text-foreground">
                2,450{" "}
                <span className="text-sm font-normal text-muted-foreground">
                  / 10,000
                </span>
              </div>
              <span className="text-xs text-muted-foreground font-mono">
                7,550 remaining
              </span>
            </div>

            {/* Progress Bar (Primary color) */}
            <div className="space-y-1.5">
              <div className="h-2.5 w-full bg-zinc-800/80 rounded-full overflow-hidden p-0.5">
                <div
                  className="h-full bg-primary rounded-full transition-all duration-500"
                  style={{ width: "24.5%" }}
                />
              </div>
              <div className="flex justify-between text-[11px] text-muted-foreground font-mono">
                <span>0 ops</span>
                <span>Limit: 10,000 ops / mo</span>
              </div>
            </div>
          </CardContent>

          <CardFooter className="border-t border-border/40 py-2.5 text-[11px] text-muted-foreground flex justify-between">
            <span>Average: ~81 ops / day</span>
            <span className="text-emerald-400">Normal velocity</span>
          </CardFooter>
        </Card>

        {/* Recall Operations Meter */}
        <Card className="border-border bg-card/70 flex flex-col justify-between">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="p-2 rounded-lg bg-blue-500/10 text-blue-400">
                  <Search className="size-4" />
                </div>
                <div>
                  <CardTitle className="text-base font-semibold">
                    Recall Operations
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Associative search & pattern completion
                  </CardDescription>
                </div>
              </div>
              <Badge
                variant="outline"
                className="font-mono text-xs border-blue-500/30 text-blue-400 bg-blue-500/5"
              >
                24.6%
              </Badge>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="flex items-baseline justify-between">
              <div className="text-2xl font-bold font-mono tracking-tight text-foreground">
                1,230{" "}
                <span className="text-sm font-normal text-muted-foreground">
                  / 5,000
                </span>
              </div>
              <span className="text-xs text-muted-foreground font-mono">
                3,770 remaining
              </span>
            </div>

            {/* Progress Bar in Blue */}
            <div className="space-y-1.5">
              <div className="h-2.5 w-full bg-zinc-800/80 rounded-full overflow-hidden p-0.5">
                <div
                  className="h-full bg-blue-500 rounded-full transition-all duration-500"
                  style={{ width: "24.6%" }}
                />
              </div>
              <div className="flex justify-between text-[11px] text-muted-foreground font-mono">
                <span>0 ops</span>
                <span>Limit: 5,000 ops / mo</span>
              </div>
            </div>
          </CardContent>

          <CardFooter className="border-t border-border/40 py-2.5 text-[11px] text-muted-foreground flex justify-between">
            <span>Average: ~41 ops / day</span>
            <span className="text-blue-400">Within threshold</span>
          </CardFooter>
        </Card>
      </div>

      {/* 3 Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
        {/* Sleep Cycles Stat Card */}
        <Card className="border-border bg-card/70 hover:bg-card hover:border-zinc-700 transition-all">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                Sleep Cycles
              </span>
              <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
                <Moon className="size-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-3xl font-bold font-mono text-foreground">
              47
            </div>
            <p className="text-xs text-muted-foreground">
              SWS Consolidation runs executed
            </p>
          </CardContent>
          <CardFooter className="pt-2 text-[11px] text-zinc-400 border-t border-border/40">
            <span className="text-emerald-400 font-medium">99.8%</span>
            &nbsp;consolidation fidelity
          </CardFooter>
        </Card>

        {/* Active Memories Stat Card */}
        <Card className="border-border bg-card/70 hover:bg-card hover:border-zinc-700 transition-all">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                Active Memories
              </span>
              <div className="p-2 rounded-lg bg-emerald-500/10 text-emerald-400">
                <Database className="size-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-3xl font-bold font-mono text-foreground">
              2,847
            </div>
            <p className="text-xs text-muted-foreground">
              Engrams consolidated in brain storage
            </p>
          </CardContent>
          <CardFooter className="pt-2 text-[11px] text-zinc-400 border-t border-border/40">
            Episodic: 1.4k | Semantic: 1.1k | Proc: 325
          </CardFooter>
        </Card>

        {/* Brain Instances Stat Card */}
        <Card className="border-border bg-card/70 hover:bg-card hover:border-zinc-700 transition-all">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                Brain Instances
              </span>
              <div className="p-2 rounded-lg bg-amber-500/10 text-amber-400">
                <Cpu className="size-4" />
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-1">
            <div className="text-3xl font-bold font-mono text-foreground">
              1 / 1
            </div>
            <p className="text-xs text-muted-foreground">
              Active neural memory instances
            </p>
          </CardContent>
          <CardFooter className="pt-2 text-[11px] text-zinc-400 border-t border-border/40">
            Starter tier limit: 1 instance
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}
