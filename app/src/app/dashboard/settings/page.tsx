"use client";

import * as React from "react";
import {
  User,
  FolderGit2,
  Brain,
  AlertTriangle,
  Lock,
  Sparkles,
  Check,
  Moon,
  Globe2,
  Trash2,
  Save,
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
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export default function SettingsPage() {
  // Profile State
  const [name, setName] = React.useState("Megha Surya");
  const [profileSaved, setProfileSaved] = React.useState(false);

  // Project State
  const [projectName, setProjectName] = React.useState("Origin Brain");
  const [defaultAgentId, setDefaultAgentId] = React.useState("demo_agent");
  const [projectSaved, setProjectSaved] = React.useState(false);

  // Brain Configuration State
  const [autoSleep, setAutoSleep] = React.useState(true);
  const [worldModel, setWorldModel] = React.useState(true);

  // Danger Zone actions
  const [clearingMemories, setClearingMemories] = React.useState(false);
  const [deletingProject, setDeletingProject] = React.useState(false);
  const [dangerNotice, setDangerNotice] = React.useState<string | null>(null);

  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    setProfileSaved(true);
    setTimeout(() => setProfileSaved(false), 2500);
  };

  const handleSaveProject = (e: React.FormEvent) => {
    e.preventDefault();
    setProjectSaved(true);
    setTimeout(() => setProjectSaved(false), 2500);
  };

  const handleClearMemories = () => {
    if (confirm("Are you sure you want to clear all memory engrams? This action is irreversible.")) {
      setClearingMemories(true);
      setTimeout(() => {
        setClearingMemories(false);
        setDangerNotice("All memory traces have been successfully cleared.");
        setTimeout(() => setDangerNotice(null), 3500);
      }, 1000);
    }
  };

  const handleDeleteProject = () => {
    if (confirm("Are you sure you want to delete this project? All instances and data will be permanently removed.")) {
      setDeletingProject(true);
      setTimeout(() => {
        setDeletingProject(false);
        setDangerNotice("Project deletion request queued.");
        setTimeout(() => setDangerNotice(null), 3500);
      }, 1000);
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6 md:p-8 max-w-5xl mx-auto w-full">
      {/* Header */}
      <div>
        <h1 className="text-2xl md:text-3xl font-semibold tracking-tight text-foreground">
          Settings
        </h1>
        <p className="text-sm text-muted-foreground mt-1">
          Manage your account profile, project defaults, neural brain architecture
          parameters, and danger zone actions.
        </p>
      </div>

      {dangerNotice && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-300 text-xs flex items-center gap-2">
          <AlertTriangle className="size-4 shrink-0" />
          <span>{dangerNotice}</span>
        </div>
      )}

      {/* 1. Profile Card */}
      <Card className="border-border bg-card/70">
        <form onSubmit={handleSaveProfile}>
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-primary/10 text-primary">
                <User className="size-4" />
              </div>
              <div>
                <CardTitle className="text-base font-semibold">
                  Profile Information
                </CardTitle>
                <CardDescription className="text-xs">
                  Your personal identity and contact credentials.
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground">
                  Display Name
                </label>
                <Input
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Your Name"
                  className="bg-background/50 h-9"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <label className="text-xs font-medium text-foreground">
                    Email Address
                  </label>
                  <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                    <Lock className="size-3" /> Managed via SSO
                  </span>
                </div>
                <Input
                  value="bodepudimegha@gmail.com"
                  disabled
                  className="bg-muted/40 text-muted-foreground cursor-not-allowed h-9"
                />
              </div>
            </div>
          </CardContent>

          <CardFooter className="flex justify-end border-t border-border/40 pt-4">
            <Button type="submit" size="sm" className="gap-1.5 cursor-pointer">
              {profileSaved ? (
                <>
                  <Check className="size-3.5 text-emerald-400" />
                  Saved
                </>
              ) : (
                <>
                  <Save className="size-3.5" />
                  Save Profile
                </>
              )}
            </Button>
          </CardFooter>
        </form>
      </Card>

      {/* 2. Project Card */}
      <Card className="border-border bg-card/70">
        <form onSubmit={handleSaveProject}>
          <CardHeader>
            <div className="flex items-center gap-2">
              <div className="p-2 rounded-lg bg-cyan-500/10 text-cyan-400">
                <FolderGit2 className="size-4" />
              </div>
              <div>
                <CardTitle className="text-base font-semibold">
                  Project Settings
                </CardTitle>
                <CardDescription className="text-xs">
                  Identify your active workspace and default agent routing identity.
                </CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground">
                  Project Name
                </label>
                <Input
                  value={projectName}
                  onChange={(e) => setProjectName(e.target.value)}
                  placeholder="Origin Brain"
                  className="bg-background/50 h-9"
                  required
                />
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-foreground">
                  Default Agent ID
                </label>
                <Input
                  value={defaultAgentId}
                  onChange={(e) => setDefaultAgentId(e.target.value)}
                  placeholder="demo_agent"
                  className="bg-background/50 font-mono text-xs h-9"
                  required
                />
              </div>
            </div>
          </CardContent>

          <CardFooter className="flex justify-end border-t border-border/40 pt-4">
            <Button type="submit" size="sm" className="gap-1.5 cursor-pointer">
              {projectSaved ? (
                <>
                  <Check className="size-3.5 text-emerald-400" />
                  Saved
                </>
              ) : (
                <>
                  <Save className="size-3.5" />
                  Save Project
                </>
              )}
            </Button>
          </CardFooter>
        </form>
      </Card>

      {/* 3. Brain Configuration Card */}
      <Card className="border-border bg-card/70">
        <CardHeader>
          <div className="flex items-center gap-2">
            <div className="p-2 rounded-lg bg-indigo-500/10 text-indigo-400">
              <Brain className="size-4" />
            </div>
            <div>
              <CardTitle className="text-base font-semibold">
                Brain Configuration
              </CardTitle>
              <CardDescription className="text-xs">
                Neuro-symbolic execution engine features and consolidation
                parameters.
              </CardDescription>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="divide-y divide-border/50 text-xs">
            {/* Neural Embeddings */}
            <div className="py-3 flex items-center justify-between gap-4">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-foreground text-sm">
                    Neural Embeddings
                  </span>
                  <Badge
                    variant="secondary"
                    className="bg-purple-500/10 text-purple-400 border border-purple-500/20 font-mono text-[10px]"
                  >
                    Pro Plan
                  </Badge>
                </div>
                <p className="text-muted-foreground">
                  High-dimensional cortical vector projections for contextual
                  invariance and semantic clustering.
                </p>
              </div>
              <Badge variant="outline" className="text-xs text-muted-foreground shrink-0">
                Locked
              </Badge>
            </div>

            {/* Auto Sleep */}
            <div className="py-3 flex items-center justify-between gap-4">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-foreground text-sm">
                    Auto Sleep
                  </span>
                  <Badge
                    variant="secondary"
                    className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono text-[10px]"
                  >
                    Enabled
                  </Badge>
                </div>
                <p className="text-muted-foreground">
                  Triggers Slow-Wave Sleep (SWS) consolidation routines
                  periodically when idle or when hippocampal buffer exceeds
                  threshold.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setAutoSleep(!autoSleep)}
                className="text-xs font-mono shrink-0 cursor-pointer"
              >
                {autoSleep ? "Disable" : "Enable"}
              </Button>
            </div>

            {/* World Model */}
            <div className="py-3 flex items-center justify-between gap-4">
              <div className="space-y-0.5">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-foreground text-sm">
                    World Model
                  </span>
                  <Badge
                    variant="secondary"
                    className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-mono text-[10px]"
                  >
                    Enabled
                  </Badge>
                </div>
                <p className="text-muted-foreground">
                  Synthesizes high-level relational ontology graphs and entity
                  state predictions across temporal interactions.
                </p>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setWorldModel(!worldModel)}
                className="text-xs font-mono shrink-0 cursor-pointer"
              >
                {worldModel ? "Disable" : "Enable"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* 4. Danger Zone Card */}
      <Card className="border-red-500/40 bg-red-950/10">
        <CardHeader>
          <div className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="size-4 shrink-0" />
            <CardTitle className="text-base font-semibold text-red-400">
              Danger Zone
            </CardTitle>
          </div>
          <CardDescription className="text-xs text-red-300/70">
            Destructive operations that cannot be undone. Please proceed with
            caution.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-3 rounded-lg border border-red-500/20 bg-red-950/20">
            <div className="space-y-0.5">
              <div className="text-sm font-medium text-foreground">
                Clear All Memories
              </div>
              <p className="text-xs text-muted-foreground">
                Permanently wipes all episodic, semantic, and procedural traces
                from the brain index.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleClearMemories}
              disabled={clearingMemories}
              className="border-red-500/50 text-red-400 hover:bg-red-500/10 hover:text-red-300 hover:border-red-500 cursor-pointer shrink-0 font-medium"
            >
              <Trash2 className="size-3.5 mr-1.5" />
              {clearingMemories ? "Clearing..." : "Clear All Memories"}
            </Button>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-3 rounded-lg border border-red-500/20 bg-red-950/20">
            <div className="space-y-0.5">
              <div className="text-sm font-medium text-foreground">
                Delete Project
              </div>
              <p className="text-xs text-muted-foreground">
                Deletes the entire Origin Brain project, API keys, and associated
                vector stores.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={handleDeleteProject}
              disabled={deletingProject}
              className="border-red-500/50 text-red-400 hover:bg-red-500/10 hover:text-red-300 hover:border-red-500 cursor-pointer shrink-0 font-medium"
            >
              <Trash2 className="size-3.5 mr-1.5" />
              {deletingProject ? "Deleting..." : "Delete Project"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
