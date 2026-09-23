import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ShieldCheck } from "lucide-react";
import { getReviewQueue, reviewCandidate } from "../lib/api";
import type { ReviewQueueItem } from "../types/saad";

export function ReviewQueue() {
  const [items, setItems] = useState<ReviewQueueItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    getReviewQueue()
      .then((queue) => { if (active) setItems(queue); })
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Cannot load review queue."); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, []);

  async function decide(item: ReviewQueueItem, action: "ACCEPT" | "REJECT") {
    setSaving(`${item.scanId}/${item.candidateId}`);
    setError(null);
    try {
      await reviewCandidate(item.scanId, item.candidateId, action);
      setItems(await getReviewQueue());
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save review.");
    } finally {
      setSaving(null);
    }
  }

  return <div className="space-y-6">
    <div className="flex items-center gap-3"><ShieldCheck className="h-5 w-5 text-cyan-400" /><h1 className="text-2xl font-semibold">Review Queue</h1></div>
    {loading && <p role="status" className="text-sm text-muted-foreground">Loading pending candidates…</p>}
    {error && <p role="alert" className="text-sm text-red-300">{error}</p>}
    {!loading && !error && items.length === 0 && <p className="text-sm text-muted-foreground">No pending candidates.</p>}
    <div className="space-y-3">
      {items.map((item) => <article key={`${item.scanId}/${item.candidateId}`} className="flex flex-col gap-3 rounded-xl border border-border/70 bg-card/40 p-4 sm:flex-row sm:items-center">
        <div className="min-w-0 flex-1">
          <Link className="font-mono text-xs text-cyan-300 hover:underline" to={`/scan/${item.scanId}`}>{item.scanId} / {item.candidateId}</Link>
          <p className="mt-1 text-[10px] text-muted-foreground">{item.prediction.priorityLevel} · priority {(item.prediction.priority * 100).toFixed(1)}% · {item.prediction.evidenceProfile}</p>
        </div>
        <div className="flex gap-2 text-xs">
          <button type="button" disabled={saving !== null} onClick={() => { void decide(item, "ACCEPT"); }} className="rounded-lg bg-emerald-400/10 px-3 py-2 text-emerald-300 disabled:opacity-50">Accept</button>
          <button type="button" disabled={saving !== null} onClick={() => { void decide(item, "REJECT"); }} className="rounded-lg bg-red-400/10 px-3 py-2 text-red-300 disabled:opacity-50">Reject</button>
          <Link className="rounded-lg border border-border px-3 py-2" to={`/scan/${item.scanId}`}>Inspect / correct</Link>
        </div>
      </article>)}
    </div>
  </div>;
}
