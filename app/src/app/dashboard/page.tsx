import Link from "next/link";
import {
  Brain,
  Layers,
  Activity,
  Moon,
  ArrowUpRight,
  Sparkles,
  Bot,
  Clock,
  ArrowRight,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

interface StatItem {
  title: string;
  value: string;
  change: string;
  icon: React.ElementType;
  description: string;
}

const stats: StatItem[] = [
  {
    title: "Total Memories",
    value: "2,847",
    change: "+342",
    icon: Brain,
    description: "Active knowledge nodes in substrate",
  },
  {
    title: "Encode Operations",
    value: "12.4K",
    change: "+18%",
    icon: Layers,
    description: "Episodic embeddings ingested",
  },
  {
    title: "Recall Operations",
    value: "8.2K",
    change: "+12%",
    icon: Activity,
    description: "Vector & associative retrievals",
  },
  {
    title: "Sleep Cycles",
    value: "47",
    change: "3 today",
    icon: Moon,
    description: "Consolidation & graph abstractions",
  },
];

interface RecentMemory {
  id: string;
  content: string;
  type: "episodic" | "semantic";
  timestamp: string;
  salience: number;
}

const recentMemories: RecentMemory[] = [
  {
    id: "mem-001",
    content:
      "Consolidated user preferences: Prefers TypeScript strict typing, concise markdown documentation, and dark mode interfaces.",
    type: "semantic",
    timestamp: "3 mins ago",
    salience: 0.95,
  },
  {
    id: "mem-002",
    content:
      "Session 104 trace: Evaluated vector clustering accuracy across 10,000 conversational turns with dynamic salience decay.",
    type: "episodic",
    timestamp: "18 mins ago",
    salience: 0.89,
  },
  {
    id: "mem-003",
    content:
      "Sleep consolidation phase #47: Pruned 412 low-salience transient tokens; abstracted 82 episodic nodes into semantic ontology.",
    type: "semantic",
    timestamp: "1 hour ago",
    salience: 0.92,
  },
  {
    id: "mem-004",
    content:
      "Observed user feedback on API latency: Sub-15ms recall achieved using local HNSW index caching with zero token drift.",
    type: "episodic",
    timestamp: "3 hours ago",
    salience: 0.78,
  },
  {
    id: "mem-005",
    content:
      "Agent initialization: Connected workspace environment C:\\Research\\OriginAI-Brain-Memory with Next.js dashboard UI.",
    type: "episodic",
    timestamp: "5 hours ago",
    salience: 0.84,
  },
];

export default function DashboardOverviewPage() {
  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Header with Title and Agent Badge */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-border/50">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              Dashboard
            </h1>
            <Badge
              variant="outline"
              className="gap-1.5 border-emerald-500/30 bg-emerald-500/10 text-emerald-400 py-0.5 px-2.5 text-xs font-medium"
            >
              <span className="size-2 rounded-full bg-emerald-400 animate-pulse" />
              <Bot className="size-3.5" />
              ox-agent-alpha (Active)
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Real-time cognitive substrate metrics and memory lifecycle overview.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Link href="/dashboard/keys">
            <Button variant="outline" size="sm" className="gap-1.5">
              API Keys
            </Button>
          </Link>
          <Link href="/dashboard/playground">
            <Button size="sm" className="gap-1.5">
              <Sparkles className="size-3.5" />
              Memory Playground
            </Button>
          </Link>
        </div>
      </div>

      {/* 4 Stat Cards in Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <Card
              key={stat.title}
              className="border-border/60 bg-card/60 backdrop-blur-sm hover:border-border transition-colors"
            >
              <CardHeader className="flex flex-row items-center justify-between pb-2 space-y-0">
                <span className="text-xs font-medium text-muted-foreground">
                  {stat.title}
                </span>
                <div className="size-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary">
                  <Icon className="size-4" />
                </div>
              </CardHeader>
              <CardContent className="space-y-1">
                <div className="text-2xl font-bold tracking-tight text-foreground">
                  {stat.value}
                </div>
                <div className="flex items-center gap-1.5 text-xs">
                  <span className="inline-flex items-center text-emerald-400 font-semibold gap-0.5">
                    <ArrowUpRight className="size-3.5" />
                    {stat.change}
                  </span>
                  <span className="text-muted-foreground text-[11px] truncate">
                    {stat.description}
                  </span>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Recent Memories Card */}
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm shadow-sm">
        <CardHeader className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-4 border-b border-border/40">
          <div>
            <CardTitle className="text-lg font-semibold flex items-center gap-2">
              <Brain className="size-5 text-primary" />
              Recent Memories
            </CardTitle>
            <CardDescription className="text-xs text-muted-foreground mt-0.5">
              Live episodic and semantic nodes processed across recent cognitive cycles
            </CardDescription>
          </div>
          <Link href="/dashboard/memories">
            <Button variant="ghost" size="sm" className="gap-1 text-xs text-primary hover:text-primary">
              View All Memories
              <ArrowRight className="size-3.5" />
            </Button>
          </Link>
        </CardHeader>

        <CardContent className="pt-4 divide-y divide-border/40">
          {recentMemories.map((memory) => {
            const isEpisodic = memory.type === "episodic";
            return (
              <div
                key={memory.id}
                className="py-3.5 first:pt-1 last:pb-1 flex flex-col md:flex-row md:items-center justify-between gap-3 group hover:bg-muted/20 px-2 rounded-lg transition-colors"
              >
                <div className="space-y-1.5 flex-1 min-w-0 pr-4">
                  <p className="text-sm text-foreground/90 leading-relaxed font-normal">
                    {memory.content}
                  </p>
                  <div className="flex items-center gap-2.5 text-xs text-muted-foreground">
                    <span className="flex items-center gap-1 font-mono text-[11px]">
                      <Clock className="size-3" />
                      {memory.timestamp}
                    </span>
                    <span className="text-border">•</span>
                    <span className="font-mono text-[11px] text-muted-foreground/80">
                      ID: {memory.id}
                    </span>
                  </div>
                </div>

                <div className="flex items-center gap-3 shrink-0">
                  <Badge
                    variant="outline"
                    className={
                      isEpisodic
                        ? "bg-purple-500/10 text-purple-400 border-purple-500/25 capitalize font-medium text-xs"
                        : "bg-cyan-500/10 text-cyan-400 border-cyan-500/25 capitalize font-medium text-xs"
                    }
                  >
                    {memory.type}
                  </Badge>

                  <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-secondary/60 border border-border/50">
                    <span className="text-[11px] text-muted-foreground">
                      Salience:
                    </span>
                    <span className="text-xs font-mono font-semibold text-foreground">
                      {memory.salience.toFixed(2)}
                    </span>
                  </div>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>
    </div>
  );
}
