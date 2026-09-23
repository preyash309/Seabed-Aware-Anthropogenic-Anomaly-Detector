import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { History as HistoryIcon, FileText } from "lucide-react";
import { listScans, resolveImageUrl } from "../lib/api";
import type { ScanSummary } from "../types/saad";

export function History() {
  const [scans, setScans] = useState<ScanSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let active = true;
    listScans()
      .then((items) => { if (active) setScans(items); })
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Cannot load scan history."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  return <div className="space-y-6">
    <div className="flex items-center gap-3"><HistoryIcon className="h-5 w-5 text-cyan-400" /><h1 className="text-2xl font-semibold">Scan History</h1></div>
    {loading && <p role="status" className="text-sm text-muted-foreground">Loading saved scans…</p>}
    {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
    {!loading && !error && scans.length === 0 && <p className="text-sm text-muted-foreground">No saved scans yet.</p>}
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
      {scans.map((scan) => <article key={scan.scanId} className="overflow-hidden rounded-2xl border border-border/70 bg-card/40">
        <img src={resolveImageUrl(scan.imageUrl)} alt={`Sonar scan ${scan.scanId}`} className="h-36 w-full object-contain bg-black/40" />
        <div className="space-y-2 p-4">
          <p className="font-mono text-xs text-cyan-300">{scan.scanId}</p>
          <p className="truncate text-xs text-muted-foreground" title={scan.filename}>{scan.filename}</p>
          <p className="text-[10px] text-muted-foreground">{new Date(scan.createdAt).toLocaleString()} · {scan.summary.totalCandidates} candidates · {scan.analysisStatus}</p>
          <div className="flex gap-3 pt-2 text-xs">
            <Link className="text-cyan-300 hover:underline" to={`/scan/${scan.scanId}`}>Open analysis</Link>
            <Link className="flex items-center gap-1 text-muted-foreground hover:text-foreground" to={`/reports?scanId=${scan.scanId}`}><FileText className="h-3 w-3" /> Report</Link>
          </div>
        </div>
      </article>)}
    </div>
  </div>;
}
