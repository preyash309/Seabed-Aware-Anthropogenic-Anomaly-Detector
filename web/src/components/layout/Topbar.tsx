import {
  Bell,
  CircleHelp,
  Cpu,
  Menu,
  Settings2,
  Wifi,
} from "lucide-react";

export function Topbar() {
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-border/70 bg-background/80 px-5 backdrop-blur-xl lg:px-7">
      <div className="flex items-center gap-3">
        <button className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground lg:hidden">
          <Menu className="h-5 w-5" />
        </button>

        <div>
          <p className="text-xs font-medium text-muted-foreground">
            Side-Scan Sonar Analysis
          </p>
          <p className="font-mono text-[10px] text-muted-foreground/60">
            MISSION / ACTIVE SESSION
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <div className="hidden items-center gap-2 rounded-md border border-border/60 bg-card/50 px-3 py-2 sm:flex">
          <Cpu className="h-3.5 w-3.5 text-cyan-400" />
          <span className="font-mono text-[10px] text-muted-foreground">
            GPU READY
          </span>
        </div>

        <div className="hidden items-center gap-2 rounded-md border border-border/60 bg-card/50 px-3 py-2 sm:flex">
          <Wifi className="h-3.5 w-3.5 text-emerald-400" />
          <span className="font-mono text-[10px] text-muted-foreground">
            LOCAL
          </span>
        </div>

        <button className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground">
          <CircleHelp className="h-4 w-4" />
        </button>

        <button className="relative rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground">
          <Bell className="h-4 w-4" />

          <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-amber-400" />
        </button>

        <button className="rounded-md p-2 text-muted-foreground hover:bg-muted hover:text-foreground">
          <Settings2 className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}