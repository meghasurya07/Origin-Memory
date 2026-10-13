"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import {
  Brain,
  BarChart3,
  Key,
  Database,
  FlaskConical,
  Settings,
  ChevronsUpDown,
  Moon,
  LogOut,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "cn";

const navItems = [
  { name: "Overview", href: "/dashboard", icon: BarChart3 },
  { name: "API Keys", href: "/dashboard/keys", icon: Key },
  { name: "Memory Browser", href: "/dashboard/memories", icon: Database },
  { name: "Playground", href: "/dashboard/playground", icon: FlaskConical },
  { name: "Usage & Billing", href: "/dashboard/usage", icon: BarChart3 },
  { name: "Settings", href: "/dashboard/settings", icon: Settings },
];

export default function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const pathname = usePathname();
  const router = useRouter();

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden text-foreground">
      {/* Left Sidebar */}
      <aside className="w-64 shrink-0 border-r border-sidebar-border bg-sidebar flex flex-col h-full select-none">
        {/* Brand Header */}
        <div className="h-14 px-4 flex items-center justify-between border-b border-sidebar-border">
          <Link href="/dashboard" className="flex items-center gap-2.5">
            <div className="size-8 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shadow-xs">
              <Brain className="size-5 text-primary" />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-base tracking-tight text-sidebar-foreground">
                OxMemory
              </span>
              <Badge
                variant="secondary"
                className="text-[10px] px-1.5 py-0 h-4 uppercase font-bold tracking-wider bg-primary/10 text-primary border-primary/20"
              >
                BETA
              </Badge>
            </div>
          </Link>
        </div>

        {/* Navigation Items */}
        <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
          <div className="px-2 pb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/70">
            Platform
          </div>
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive =
              item.href === "/dashboard"
                ? pathname === "/dashboard"
                : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground shadow-xs"
                    : "text-sidebar-foreground/70 hover:bg-sidebar-accent/50 hover:text-sidebar-foreground"
                )}
              >
                <Icon
                  className={cn(
                    "size-4 shrink-0",
                    isActive ? "text-primary" : "text-muted-foreground"
                  )}
                />
                <span className="truncate">{item.name}</span>
              </Link>
            );
          })}
        </div>

        {/* Usage Indicator Bar */}
        <div className="px-3 py-2">
          <div className="rounded-xl border border-sidebar-border bg-sidebar-accent/30 p-3">
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-medium text-sidebar-foreground">
                Encode Operations
              </span>
              <span className="text-muted-foreground font-mono text-[11px]">
                24.5%
              </span>
            </div>
            <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
              <div
                className="h-full rounded-full bg-primary transition-all duration-500"
                style={{ width: "24.5%" }}
              />
            </div>
            <div className="mt-2 flex items-center justify-between text-[11px] text-muted-foreground">
              <span>2,450 / 10,000 ops</span>
              <Link
                href="/dashboard/usage"
                className="font-medium text-primary hover:underline"
              >
                Upgrade
              </Link>
            </div>
          </div>
        </div>

        {/* User Dropdown at bottom */}
        <div className="p-3 border-t border-sidebar-border mt-auto">
          <DropdownMenu>
            <DropdownMenuTrigger className="flex w-full items-center gap-3 rounded-lg p-2 text-left hover:bg-sidebar-accent transition-colors outline-none cursor-pointer">
              <Avatar size="sm">
                <AvatarFallback className="bg-primary/20 text-primary font-semibold text-xs">
                  MS
                </AvatarFallback>
              </Avatar>
              <div className="flex flex-col min-w-0 flex-1 leading-tight">
                <span className="text-sm font-medium truncate text-sidebar-foreground">
                  Megha Surya
                </span>
                <span className="text-xs text-muted-foreground truncate">
                  Free Plan
                </span>
              </div>
              <ChevronsUpDown className="size-4 text-muted-foreground shrink-0" />
            </DropdownMenuTrigger>
            <DropdownMenuContent
              side="top"
              align="start"
              className="w-56 mb-2 bg-popover text-popover-foreground shadow-xl border border-border"
            >
              <DropdownMenuLabel className="font-normal">
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium leading-none">
                    Megha Surya
                  </p>
                  <p className="text-xs leading-none text-muted-foreground">
                    megha.surya@originai.dev
                  </p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => router.push("/dashboard/settings")}
              >
                <Settings className="size-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Moon className="size-4 mr-2" />
                Dark Theme
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                variant="destructive"
                onClick={() => router.push("/login")}
              >
                <LogOut className="size-4 mr-2" />
                Sign Out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 h-screen overflow-hidden">
        <main className="flex-1 overflow-y-auto bg-background p-6 lg:p-8">
          {children}
        </main>
      </div>
    </div>
  );
}
