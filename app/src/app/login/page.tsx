"use client";

import Link from "next/link";
import { Brain, GitFork, Mail, ArrowRight } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function LoginPage() {
  return (
    <div className="relative min-h-screen w-full flex flex-col items-center justify-center p-4 bg-background overflow-hidden">
      {/* Background subtle glowing radial gradients */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[350px] bg-primary/10 rounded-full blur-3xl pointer-events-none -z-10" />
      <div className="absolute bottom-1/4 left-1/3 w-[400px] h-[250px] bg-blue-500/5 rounded-full blur-3xl pointer-events-none -z-10" />

      <div className="w-full max-w-md space-y-6">
        <Card className="border-border/60 bg-card/90 shadow-2xl backdrop-blur-xl">
          <CardHeader className="text-center pb-4 pt-6 space-y-2">
            <div className="mx-auto flex size-12 items-center justify-center rounded-xl bg-primary/10 border border-primary/25 shadow-inner">
              <Brain className="size-6 text-primary" />
            </div>
            <CardTitle className="text-2xl font-bold tracking-tight">
              OxMemory
            </CardTitle>
            <CardDescription className="text-sm text-muted-foreground">
              Autonomous cognitive substrate & long-term memory for AI agents
            </CardDescription>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* OAuth Buttons */}
            <div className="space-y-2.5">
              <Link href="/dashboard" className="block w-full">
                <Button
                  variant="outline"
                  type="button"
                  className="w-full h-10 gap-2.5 border-border/80 hover:bg-muted font-medium text-sm transition-colors"
                >
                  <GitFork className="size-4" />
                  Continue with GitHub
                </Button>
              </Link>
              <Link href="/dashboard" className="block w-full">
                <Button
                  variant="outline"
                  type="button"
                  className="w-full h-10 gap-2.5 border-border/80 hover:bg-muted font-medium text-sm transition-colors"
                >
                  <Mail className="size-4" />
                  Continue with Google
                </Button>
              </Link>
            </div>

            {/* Divider */}
            <div className="relative my-4">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-border/70" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-card px-3 text-muted-foreground font-medium">
                  or continue with email
                </span>
              </div>
            </div>

            {/* Email Login Form */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                window.location.href = "/dashboard";
              }}
              className="space-y-3"
            >
              <div className="space-y-1.5">
                <label
                  htmlFor="email"
                  className="text-xs font-medium text-foreground/80"
                >
                  Work Email
                </label>
                <Input
                  id="email"
                  type="email"
                  placeholder="agent.architect@originai.dev"
                  defaultValue="megha.surya@originai.dev"
                  className="h-10 bg-background/60"
                  required
                />
              </div>

              <Link href="/dashboard" className="block w-full pt-1">
                <Button
                  type="button"
                  className="w-full h-10 gap-2 font-medium shadow-md transition-all"
                >
                  Sign In
                  <ArrowRight className="size-4" />
                </Button>
              </Link>
            </form>
          </CardContent>

          <CardFooter className="flex flex-col gap-3 pt-1 pb-6 text-center text-xs text-muted-foreground border-t-0">
            <div>
              Don&apos;t have an account?{" "}
              <Link
                href="/dashboard"
                className="font-medium text-primary hover:underline"
              >
                Sign up free
              </Link>
            </div>
          </CardFooter>
        </Card>

        {/* Legal & Terms footer */}
        <p className="text-center text-xs text-muted-foreground/70 px-6 leading-relaxed">
          By signing in, you agree to our{" "}
          <Link
            href="/dashboard"
            className="underline underline-offset-4 hover:text-foreground transition-colors"
          >
            Terms of Service
          </Link>{" "}
          and{" "}
          <Link
            href="/dashboard"
            className="underline underline-offset-4 hover:text-foreground transition-colors"
          >
            Privacy Policy
          </Link>
          .
        </p>
      </div>
    </div>
  );
}
