"use client";

import { useState } from "react";
import {
  Key,
  Plus,
  Eye,
  EyeOff,
  Copy,
  Check,
  Trash2,
  ShieldCheck,
  Calendar,
  Clock,
  Sparkles,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";

interface ApiKey {
  id: string;
  name: string;
  key: string;
  permission: "Read & Write" | "Read Only";
  created: string;
  lastUsed: string;
}

const initialKeys: ApiKey[] = [
  {
    id: "key-prod",
    name: "Production",
    key: "ox_prod_sk_a1b2c3d4e5f6g7h8i9j0",
    permission: "Read & Write",
    created: "Oct 01, 2026",
    lastUsed: "2 mins ago",
  },
  {
    id: "key-dev",
    name: "Development",
    key: "ox_dev_sk_f9e8d7c6b5a412345678",
    permission: "Read & Write",
    created: "Sep 24, 2026",
    lastUsed: "1 hour ago",
  },
  {
    id: "key-ci",
    name: "CI/CD Pipeline",
    key: "ox_test_sk_99887766554433221100",
    permission: "Read Only",
    created: "Sep 12, 2026",
    lastUsed: "Yesterday",
  },
];

export default function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKey[]>(initialKeys);
  const [visibleKeyIds, setVisibleKeyIds] = useState<Set<string>>(new Set());
  const [copiedKeyId, setCopiedKeyId] = useState<string | null>(null);

  // Dialog state
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newKeyName, setNewKeyName] = useState("");
  const [newPermission, setNewPermission] = useState<"Read & Write" | "Read Only">("Read & Write");

  const toggleVisibility = (id: string) => {
    setVisibleKeyIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleCopy = (id: string, text: string) => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      navigator.clipboard.writeText(text);
    }
    setCopiedKeyId(id);
    setTimeout(() => {
      setCopiedKeyId(null);
    }, 2000);
  };

  const handleDelete = (id: string) => {
    setKeys((prev) => prev.filter((k) => k.id !== id));
    setVisibleKeyIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const handleCreateKey = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newKeyName.trim()) return;

    const randomSuffix =
      Math.random().toString(36).substring(2, 8) +
      Math.random().toString(36).substring(2, 8);

    const createdKey: ApiKey = {
      id: `key-${Date.now()}`,
      name: newKeyName.trim(),
      key: `ox_live_sk_${randomSuffix}`,
      permission: newPermission,
      created: "Just now",
      lastUsed: "Never",
    };

    setKeys((prev) => [createdKey, ...prev]);
    setNewKeyName("");
    setNewPermission("Read & Write");
    setIsDialogOpen(false);
  };

  const getMaskedKey = (fullKey: string) => {
    if (fullKey.length <= 16) return fullKey;
    const prefix = fullKey.substring(0, 15);
    const suffix = fullKey.substring(fullKey.length - 2);
    return `${prefix}...${suffix}`;
  };

  return (
    <div className="space-y-8 max-w-7xl mx-auto">
      {/* Page Header with Action */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-border/50">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-3xl font-bold tracking-tight text-foreground">
              API Keys
            </h1>
            <Badge
              variant="outline"
              className="gap-1 border-primary/30 bg-primary/10 text-primary py-0.5 px-2.5 text-xs font-medium"
            >
              <Key className="size-3" />
              {keys.length} Active {keys.length === 1 ? "Key" : "Keys"}
            </Badge>
          </div>
          <p className="text-sm text-muted-foreground mt-1">
            Manage authentication credentials for agent runtime access to OxMemory substrate.
          </p>
        </div>

        {/* Dialog for Creating New Key */}
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger
            render={
              <Button className="gap-2 shadow-sm">
                <Plus className="size-4" />
                Create Key
              </Button>
            }
          />

          <DialogContent className="sm:max-w-md">
            <form onSubmit={handleCreateKey}>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2">
                  <Sparkles className="size-5 text-primary" />
                  Generate New API Key
                </DialogTitle>
                <DialogDescription>
                  Create an encrypted authentication token for your agent frameworks.
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 py-4">
                <div className="space-y-1.5">
                  <label
                    htmlFor="keyName"
                    className="text-xs font-semibold uppercase tracking-wider text-muted-foreground"
                  >
                    Key Name
                  </label>
                  <Input
                    id="keyName"
                    placeholder="e.g. Production Cluster, Agent AutoGPT"
                    value={newKeyName}
                    onChange={(e) => setNewKeyName(e.target.value)}
                    required
                    className="h-9"
                    autoFocus
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    Access Permission
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setNewPermission("Read & Write")}
                      className={`px-3 py-2 rounded-lg text-xs font-medium border text-left transition-all cursor-pointer ${
                        newPermission === "Read & Write"
                          ? "border-primary bg-primary/10 text-primary shadow-xs"
                          : "border-border hover:bg-muted text-muted-foreground"
                      }`}
                    >
                      <div className="font-semibold">Read & Write</div>
                      <div className="text-[11px] opacity-80 mt-0.5">Encode & Recall</div>
                    </button>
                    <button
                      type="button"
                      onClick={() => setNewPermission("Read Only")}
                      className={`px-3 py-2 rounded-lg text-xs font-medium border text-left transition-all cursor-pointer ${
                        newPermission === "Read Only"
                          ? "border-primary bg-primary/10 text-primary shadow-xs"
                          : "border-border hover:bg-muted text-muted-foreground"
                      }`}
                    >
                      <div className="font-semibold">Read Only</div>
                      <div className="text-[11px] opacity-80 mt-0.5">Recall queries only</div>
                    </button>
                  </div>
                </div>
              </div>

              <DialogFooter className="gap-2 sm:gap-0">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsDialogOpen(false)}
                >
                  Cancel
                </Button>
                <Button type="submit" disabled={!newKeyName.trim()}>
                  Generate Key
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      {/* API Keys List Card */}
      <Card className="border-border/60 bg-card/70 backdrop-blur-sm shadow-sm">
        <CardHeader className="pb-3 border-b border-border/40">
          <CardTitle className="text-lg font-semibold">Secret Keys</CardTitle>
          <CardDescription className="text-xs text-muted-foreground">
            Keep your API keys safe. Do not expose keys in client-side code or public repositories.
          </CardDescription>
        </CardHeader>

        <CardContent className="pt-2 p-0">
          {keys.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground text-sm">
              No API keys found. Click &quot;Create Key&quot; to generate one.
            </div>
          ) : (
            <div className="divide-y divide-border/40">
              {keys.map((apiKey) => {
                const isVisible = visibleKeyIds.has(apiKey.id);
                const isCopied = copiedKeyId === apiKey.id;
                const isReadWrite = apiKey.permission === "Read & Write";

                return (
                  <div
                    key={apiKey.id}
                    className="p-4 sm:p-5 flex flex-col lg:flex-row lg:items-center justify-between gap-4 hover:bg-muted/20 transition-colors"
                  >
                    {/* Left: Name and Metadata */}
                    <div className="space-y-2 min-w-0 flex-1">
                      <div className="flex items-center gap-2.5 flex-wrap">
                        <span className="font-semibold text-foreground text-base tracking-tight">
                          {apiKey.name}
                        </span>
                        <Badge
                          variant="outline"
                          className={
                            isReadWrite
                              ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/30 text-xs font-medium"
                              : "bg-blue-500/10 text-blue-400 border-blue-500/30 text-xs font-medium"
                          }
                        >
                          <ShieldCheck className="size-3 mr-1" />
                          {apiKey.permission}
                        </Badge>
                      </div>

                      {/* Key Box with Masking */}
                      <div className="flex items-center gap-2 max-w-lg">
                        <div className="font-mono text-xs bg-muted/60 px-3 py-1.5 rounded-md border border-border/70 text-foreground/90 select-all tracking-wider truncate">
                          {isVisible ? apiKey.key : getMaskedKey(apiKey.key)}
                        </div>

                        {/* Visibility Toggle */}
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          type="button"
                          onClick={() => toggleVisibility(apiKey.id)}
                          title={isVisible ? "Hide API key" : "Show API key"}
                          className="text-muted-foreground hover:text-foreground"
                        >
                          {isVisible ? (
                            <EyeOff className="size-3.5" />
                          ) : (
                            <Eye className="size-3.5" />
                          )}
                          <span className="sr-only">Toggle Key Visibility</span>
                        </Button>

                        {/* Copy Button */}
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          type="button"
                          onClick={() => handleCopy(apiKey.id, apiKey.key)}
                          title="Copy API key"
                          className={
                            isCopied
                              ? "text-emerald-400 hover:text-emerald-400"
                              : "text-muted-foreground hover:text-foreground"
                          }
                        >
                          {isCopied ? (
                            <Check className="size-3.5 text-emerald-400" />
                          ) : (
                            <Copy className="size-3.5" />
                          )}
                          <span className="sr-only">Copy Key</span>
                        </Button>
                      </div>

                      {/* Timestamps */}
                      <div className="flex items-center gap-4 text-xs text-muted-foreground pt-0.5">
                        <span className="flex items-center gap-1.5">
                          <Calendar className="size-3.5" />
                          Created: {apiKey.created}
                        </span>
                        <span>•</span>
                        <span className="flex items-center gap-1.5">
                          <Clock className="size-3.5" />
                          Last used: {apiKey.lastUsed}
                        </span>
                      </div>
                    </div>

                    {/* Right: Actions */}
                    <div className="flex items-center gap-2 self-start lg:self-center shrink-0">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleDelete(apiKey.id)}
                        className="text-red-400 hover:text-red-300 hover:bg-red-500/10 gap-1.5 text-xs"
                      >
                        <Trash2 className="size-3.5 text-red-400" />
                        Delete
                      </Button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
