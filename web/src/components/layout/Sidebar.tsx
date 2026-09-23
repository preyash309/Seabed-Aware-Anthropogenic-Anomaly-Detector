import {
  Activity,
  BarChart3,
  FileText,
  History,
  LayoutDashboard,
  ScanLine,
  Settings,
  ShieldCheck,
  Waves,
} from "lucide-react";

import { NavLink } from "react-router-dom";

const navigation = [
  {
    section: "Workspace",
    items: [
      {
        label: "Overview",
        icon: LayoutDashboard,
        path: "/",
      },
      {
        label: "New Scan",
        icon: ScanLine,
        path: "/scan",
      },
      {
        label: "Scan History",
        icon: History,
        path: "/history",
      },
    ],
  },
  {
    section: "Review",
    items: [
      {
        label: "Review Queue",
        icon: ShieldCheck,
        path: "/review",
      },
    ],
  },
  {
    section: "Reporting",
    items: [
      {
        label: "Reports",
        icon: FileText,
        path: "/reports",
      },
      {
        label: "Analytics",
        icon: BarChart3,
        path: "/analytics",
      },
    ],
  },
];

export function Sidebar() {
  return (
    <aside className="hidden w-[250px] shrink-0 border-r border-border/70 bg-card/30 lg:flex lg:flex-col">
      <div className="flex h-16 items-center border-b border-border/70 px-5">
        <NavLink to="/" className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-400/20 bg-cyan-400/5">
            <Waves className="h-5 w-5 text-cyan-400" />
          </div>

          <div>
            <p className="text-sm font-semibold tracking-wide">
              SAAD
            </p>

            <p className="text-[9px] uppercase tracking-[0.18em] text-muted-foreground">
              Marine Intelligence
            </p>
          </div>
        </NavLink>
      </div>

      <nav className="flex-1 space-y-7 p-4">
        {navigation.map((group) => (
          <div key={group.section}>
            <p className="mb-2 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground/60">
              {group.section}
            </p>

            <div className="space-y-1">
              {group.items.map((item) => {
                const Icon = item.icon;

                return (
                  <NavLink
                    key={item.label}
                    to={item.path}
                    className={({ isActive }) =>
                      `group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-sm transition-colors ${
                        isActive
                          ? "bg-cyan-400/10 text-cyan-300"
                          : "text-muted-foreground hover:bg-muted/60 hover:text-foreground"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <Icon
                          className={`h-4 w-4 ${
                            isActive
                              ? "text-cyan-400"
                              : "text-muted-foreground group-hover:text-foreground"
                          }`}
                        />

                        <span className="flex-1">
                          {item.label}
                        </span>

                      </>
                    )}
                  </NavLink>
                );
              })}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-border/70 p-4">
        <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground">
          <Settings className="h-4 w-4" />
          <span>System settings</span>
        </button>

        <div className="mt-4 rounded-lg border border-emerald-400/10 bg-emerald-400/5 p-3">
          <div className="flex items-center gap-2">
            <Activity className="h-3.5 w-3.5 text-emerald-400" />

            <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-300">
              Local analysis workspace
            </span>
          </div>

          <p className="mt-2 font-mono text-[9px] text-muted-foreground">
            SAAD • WORKSPACE
          </p>
        </div>
      </div>
    </aside>
  );
}
