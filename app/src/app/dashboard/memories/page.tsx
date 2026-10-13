"use client";

import * as React from "react";
import {
  Search,
  Filter,
  Trash2,
  Brain,
  Clock,
  Sparkles,
  Calendar,
  Layers,
  ChevronDown,
  RefreshCw,
  Hash,
} from "lucide-react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button, buttonVariants } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";

type MemoryType = "episodic" | "semantic" | "procedural";

interface MemoryItem {
  id: string;
  content: string;
  type: MemoryType;
  salience: number;
  retrievability: number;
  agent_id: string;
  createdAt: string;
  lastAccessed: string;
  accessCount: number;
}

const INITIAL_MEMORIES: MemoryItem[] = [
  {
    id: "mem-001",
    content:
      "User discussed preferring async Python architecture over multithreading for the brain memory retrieval pipeline to reduce lock contention.",
    type: "episodic",
    salience: 0.92,
    retrievability: 0.88,
    agent_id: "agent_alpha",
    createdAt: "2026-10-12 14:22:10",
    lastAccessed: "10 mins ago",
    accessCount: 14,
  },
  {
    id: "mem-002",
    content:
      "OriginAI Brain integrates biologically inspired Dual-Store architecture: Hippocampus fast-learning index and Neocortex long-term consolidated store.",
    type: "semantic",
    salience: 0.95,
    retrievability: 0.91,
    agent_id: "system_core",
    createdAt: "2026-10-10 09:15:44",
    lastAccessed: "2 hours ago",
    accessCount: 89,
  },
  {
    id: "mem-003",
    content:
      "Sleep consolidation protocol: 1. Sample high-salience episodic traces. 2. Replay patterns with noise injection. 3. Backprop into neocortex weights. 4. Prune decayed synapses.",
    type: "procedural",
    salience: 0.84,
    retrievability: 0.79,
    agent_id: "worker_daemon",
    createdAt: "2026-10-08 18:30:00",
    lastAccessed: "Yesterday",
    accessCount: 32,
  },
  {
    id: "mem-004",
    content:
      "Session log: Debugged vector similarity threshold calibration; adjusted cosine cutoff from 0.75 to 0.82 for high-precision recall in domain queries.",
    type: "episodic",
    salience: 0.78,
    retrievability: 0.65,
    agent_id: "agent_alpha",
    createdAt: "2026-10-11 21:04:12",
    lastAccessed: "5 hours ago",
    accessCount: 7,
  },
  {
    id: "mem-005",
    content:
      "Synaptic plasticity decay follows Ebbinghaus power law modified with neuromodulatory dopamine and acetylcholine scaling factors.",
    type: "semantic",
    salience: 0.89,
    retrievability: 0.83,
    agent_id: "system_core",
    createdAt: "2026-10-09 11:45:20",
    lastAccessed: "1 day ago",
    accessCount: 45,
  },
  {
    id: "mem-006",
    content:
      "Engram reconstruction routine: Activate hippocampal seed node -> traverse associative graph within 2-hop radius -> assemble reconstructed context token stream.",
    type: "procedural",
    salience: 0.91,
    retrievability: 0.86,
    agent_id: "worker_daemon",
    createdAt: "2026-10-07 16:10:05",
    lastAccessed: "3 days ago",
    accessCount: 23,
  },
];

export default function MemoriesPage() {
  const [memories, setMemories] = React.useState<MemoryItem[]>(INITIAL_MEMORIES);
  const [searchQuery, setSearchQuery] = React.useState("");
  const [typeFilter, setTypeFilter] = React.useState<"all" | MemoryType>("all");

  const filteredMemories = React.useMemo(() => {
    return memories.filter((mem) => {
      const matchesSearch =
        mem.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
        mem.agent_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
        mem.id.toLowerCase().includes(searchQuery.toLowerCase());

      const matchesType = typeFilter === "all" || mem.type === typeFilter;

      return matchesSearch && matchesType;
    });
  }, [memories, searchQuery, typeFilter]);

  const handleDelete = (id: string) => {
    setMemories((prev) => prev.filter((item) => item.id !== id));
  };

  const handleReset = () => {
    setMemories(INITIAL_MEMORIES);
    setSearchQuery("");
    setTypeFilter("all");
  };

  const getTypeBadge = (type: MemoryType) => {
    switch (type) {
      case "episodic":
        return (
          <Badge
            variant="secondary"
            className="bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-mono capitalize"
          >
            Episodic
          </Badge>
        );
      case "semantic":
        return (
          <Badge
            variant="secondary"
            className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono capitalize"
          >
            Semantic
          </Badge>
        );
      case "procedural":
        return (
          <Badge
            variant="secondary"
            className="bg-amber-500/10 text-amber-400 border border-amber-500/20 font-mono capitalize"
          >
            Procedural
          </Badge>
        );
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 max-w-7xl mx-auto w-full">
      {/* Header section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">
              Memory Browser
            </h1>
            <Badge variant="outline" className="text-xs font-mono">
              {filteredMemories.length} / {memories.length} Loaded
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Explore, inspect, and manage engrams stored across your episodic,
            semantic, and procedural memory stores.
          </p>
        </div>

        {memories.length < INITIAL_MEMORIES.length && (
          <Button
            variant="outline"
            size="sm"
            onClick={handleReset}
            className="self-start md:self-auto gap-1.5"
          >
            <RefreshCw className="size-3.5" />
            Reset Mock Data
          </Button>
        )}
      </div>

      {/* Filter Toolbar */}
      <Card className="bg-card/60 backdrop-blur-sm border-border">
        <CardContent className="p-4 flex flex-col sm:flex-row gap-3 items-stretch sm:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
            <Input
              placeholder="Search memory content, agent ID, or memory ID..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-9 bg-background/50 h-9"
            />
          </div>

          <div className="flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger
                className={buttonVariants({
                  variant: "outline",
                  size: "sm",
                  className: "gap-2 min-w-[140px] justify-between cursor-pointer",
                })}
              >
                <span className="flex items-center gap-1.5 capitalize">
                  <Filter className="size-3.5 text-muted-foreground" />
                  {typeFilter === "all" ? "All types" : typeFilter}
                </span>
                <ChevronDown className="size-3.5 opacity-60" />
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-44">
                <DropdownMenuLabel>Filter by Memory Type</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => setTypeFilter("all")}>
                  All types
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTypeFilter("episodic")}>
                  Episodic
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTypeFilter("semantic")}>
                  Semantic
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setTypeFilter("procedural")}>
                  Procedural
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {(searchQuery || typeFilter !== "all") && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setSearchQuery("");
                  setTypeFilter("all");
                }}
                className="text-xs text-muted-foreground hover:text-foreground"
              >
                Clear
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Memory List */}
      <div className="flex flex-col gap-4">
        {filteredMemories.length === 0 ? (
          <Card className="bg-card/40 border-dashed border-border py-12 text-center">
            <CardContent className="flex flex-col items-center justify-center gap-2">
              <Brain className="size-8 text-muted-foreground/60" />
              <p className="text-base font-medium text-foreground">
                No memories match your query
              </p>
              <p className="text-xs text-muted-foreground max-w-sm">
                Try loosening your search terms or changing the memory type
                filter above.
              </p>
              <Button
                variant="outline"
                size="sm"
                className="mt-2"
                onClick={() => {
                  setSearchQuery("");
                  setTypeFilter("all");
                }}
              >
                Reset Filters
              </Button>
            </CardContent>
          </Card>
        ) : (
          filteredMemories.map((mem) => {
            const saliencePercent = Math.round(mem.salience * 100);
            const retrievabilityPercent = Math.round(mem.retrievability * 100);

            return (
              <Card
                key={mem.id}
                className="group relative border-border bg-card/70 hover:bg-card hover:border-zinc-700 transition-all duration-150"
              >
                <CardHeader className="pb-2">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      {getTypeBadge(mem.type)}
                      <span className="font-mono text-xs text-muted-foreground">
                        {mem.id}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono text-zinc-400 bg-zinc-800/60 px-2 py-0.5 rounded border border-zinc-700/50">
                        agent: {mem.agent_id}
                      </span>
                      <Button
                        variant="ghost"
                        size="icon-xs"
                        onClick={() => handleDelete(mem.id)}
                        className="text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                        title="Delete memory engram"
                      >
                        <Trash2 className="size-3.5" />
                        <span className="sr-only">Delete</span>
                      </Button>
                    </div>
                  </div>
                </CardHeader>

                <CardContent className="flex flex-col gap-4 pt-1">
                  {/* Content */}
                  <p className="text-sm text-foreground/90 leading-relaxed break-words font-sans">
                    {mem.content}
                  </p>

                  {/* Metadata & Scores Grid */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-border/60 text-xs">
                    {/* Salience */}
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Sparkles className="size-3 text-amber-400" />
                          Salience
                        </span>
                        <span className="font-mono font-medium text-foreground">
                          {mem.salience.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-amber-400/90 rounded-full transition-all"
                          style={{ width: `${saliencePercent}%` }}
                        />
                      </div>
                    </div>

                    {/* Retrievability */}
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center justify-between text-muted-foreground">
                        <span className="flex items-center gap-1">
                          <Layers className="size-3 text-cyan-400" />
                          Retrievability
                        </span>
                        <span className="font-mono font-medium text-foreground">
                          {mem.retrievability.toFixed(2)}
                        </span>
                      </div>
                      <div className="h-1.5 w-full bg-zinc-800 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-cyan-400/90 rounded-full transition-all"
                          style={{ width: `${retrievabilityPercent}%` }}
                        />
                      </div>
                    </div>

                    {/* Created & Last Accessed */}
                    <div className="flex flex-col justify-center text-muted-foreground">
                      <span className="flex items-center gap-1 text-[11px]">
                        <Calendar className="size-3" />
                        Created:{" "}
                        <span className="font-mono text-zinc-300">
                          {mem.createdAt.split(" ")[0]}
                        </span>
                      </span>
                      <span className="flex items-center gap-1 text-[11px] mt-0.5">
                        <Clock className="size-3" />
                        Active:{" "}
                        <span className="text-zinc-300">
                          {mem.lastAccessed}
                        </span>
                      </span>
                    </div>

                    {/* Access Count */}
                    <div className="flex flex-col justify-center sm:items-end text-muted-foreground">
                      <span className="flex items-center gap-1 text-[11px]">
                        <Hash className="size-3" />
                        Access Count:{" "}
                        <span className="font-mono font-medium text-foreground">
                          {mem.accessCount}
                        </span>
                      </span>
                      <span className="text-[10px] text-zinc-500">
                        Hippocampal trace
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })
        )}
      </div>
    </div>
  );
}
