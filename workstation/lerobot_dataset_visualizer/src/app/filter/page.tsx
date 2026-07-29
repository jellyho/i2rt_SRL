"use client";

import Link from "next/link";
import { FormEvent, useEffect, useMemo, useState } from "react";

type Dataset = { name: string; episodes: number; frames: number; fps: number };
type JobStatus = {
  jobId: string;
  state: "queued" | "running" | "complete" | "failed";
  phase: string;
  message: string;
  source: string;
  output: string;
  currentEpisode?: number;
  totalEpisodes?: number;
  keptFrames?: number;
  removedFrames?: number;
  processedFrames?: number;
  totalSourceFrames?: number;
  currentFrame?: number;
  currentEpisodeFrames?: number;
};

const SETTINGS_KEY = "i2rt-idle-timeline-settings";
const ACTIVE_JOB_KEY = "i2rt-active-filter-job";

export default function FilterDatasetPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [source, setSource] = useState("");
  const [output, setOutput] = useState("");
  const [threshold, setThreshold] = useState("0.001");
  const [minimumRun, setMinimumRun] = useState(7);
  const [idleMode, setIdleMode] = useState("both");
  const [outcomeScope, setOutcomeScope] = useState("both");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get(
      "dataset",
    );
    fetch("/api/local-datasets", { cache: "no-store" })
      .then((response) => response.json())
      .then((body: { datasets?: Dataset[] }) => {
        const available = body.datasets || [];
        setDatasets(available);
        const selected =
          (requested &&
            available.some((dataset) => dataset.name === requested) &&
            requested) ||
          available[0]?.name ||
          "";
        setSource(selected);
        setOutput(selected ? `${selected}_idle_filtered` : "");
      })
      .catch((reason) => setError(String(reason)));
    try {
      const activeJob = window.localStorage.getItem(ACTIVE_JOB_KEY);
      if (activeJob) setJobId(activeJob);
      const saved = JSON.parse(
        window.localStorage.getItem(SETTINGS_KEY) || "null",
      ) as { threshold?: unknown; minFrames?: unknown } | null;
      if (saved) {
        const savedThreshold = Number(saved.threshold);
        const savedMinimum = Number(saved.minFrames);
        if (Number.isFinite(savedThreshold) && savedThreshold > 0)
          setThreshold(String(savedThreshold));
        if (Number.isInteger(savedMinimum) && savedMinimum > 0)
          setMinimumRun(savedMinimum);
      }
    } catch {
      // Keep defaults when browser storage is unavailable or malformed.
    }
  }, []);

  useEffect(() => {
    if (!jobId) return;
    let stopped = false;
    const poll = async () => {
      try {
        const response = await fetch(`/api/filter-dataset?job_id=${jobId}`, {
          cache: "no-store",
        });
        const body = (await response.json()) as JobStatus & { error?: string };
        if (!response.ok)
          throw new Error(body.error || `HTTP ${response.status}`);
        if (!stopped) setStatus(body);
      } catch (reason) {
        if (!stopped)
          setError(reason instanceof Error ? reason.message : String(reason));
      }
    };
    void poll();
    const timer = window.setInterval(poll, 1500);
    return () => {
      stopped = true;
      window.clearInterval(timer);
    };
  }, [jobId]);

  const selectedDataset = datasets.find((dataset) => dataset.name === source);
  const progress = useMemo(() => {
    if (status?.totalSourceFrames) {
      return Math.min(
        100,
        ((status.processedFrames || 0) / status.totalSourceFrames) * 100,
      );
    }
    if (!status?.totalEpisodes) return 0;
    return Math.min(
      100,
      ((status.currentEpisode || 0) / status.totalEpisodes) * 100,
    );
  }, [status]);
  const running =
    status?.state === "queued" ||
    status?.state === "running" ||
    (!!jobId && !status);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setError(null);
    setStatus(null);
    try {
      const response = await fetch("/api/filter-dataset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source,
          output,
          threshold: Number(threshold),
          minimumRun,
          idleMode,
          outcomeScope,
        }),
      });
      const body = (await response.json()) as {
        jobId?: string;
        error?: string;
      };
      if (!response.ok || !body.jobId)
        throw new Error(body.error || `HTTP ${response.status}`);
      setJobId(body.jobId);
      window.localStorage.setItem(ACTIVE_JOB_KEY, body.jobId);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  return (
    <main className="min-h-screen bg-[var(--bg)] px-6 py-10 text-slate-100">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-start justify-between gap-4 border-b border-white/10 pb-6">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.22em] text-cyan-400">
              i2rt · offline
            </p>
            <h1 className="mt-2 text-3xl font-semibold">
              Create idle-filtered dataset
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Rebuild synchronized Parquet and video into a new dataset. The
              source is never modified.
            </p>
          </div>
          <Link href="/" className="text-sm text-slate-400 hover:text-cyan-300">
            Datasets
          </Link>
        </div>

        <form onSubmit={submit} className="mt-7 space-y-6">
          <section className="panel grid gap-4 p-5 sm:grid-cols-2">
            <label className="text-xs text-slate-400">
              Source dataset
              <select
                value={source}
                disabled={running}
                onChange={(event) => {
                  setSource(event.target.value);
                  setOutput(`${event.target.value}_idle_filtered`);
                }}
                className="mt-2 block w-full rounded border border-white/10 bg-[var(--surface-1)] px-3 py-2 text-sm text-slate-200"
              >
                {datasets.map((dataset) => (
                  <option key={dataset.name}>{dataset.name}</option>
                ))}
              </select>
              {selectedDataset && (
                <span className="mt-1.5 block text-[10px] text-slate-600">
                  {selectedDataset.episodes} episodes ·{" "}
                  {selectedDataset.frames.toLocaleString()} frames ·{" "}
                  {selectedDataset.fps} Hz
                </span>
              )}
            </label>
            <label className="text-xs text-slate-400">
              New dataset name
              <input
                value={output}
                disabled={running}
                onChange={(event) => setOutput(event.target.value)}
                className="mt-2 block w-full rounded border border-white/10 bg-black/20 px-3 py-2 font-mono text-sm text-slate-200 outline-none focus:border-cyan-400/50"
              />
            </label>
          </section>

          <section className="panel grid gap-4 p-5 sm:grid-cols-2">
            <label className="text-xs text-slate-400">
              Action-change threshold
              <input
                value={threshold}
                disabled={running}
                onChange={(event) => setThreshold(event.target.value)}
                inputMode="decimal"
                className="mt-2 block w-full rounded border border-white/10 bg-black/20 px-3 py-2 font-mono text-sm text-slate-200 outline-none focus:border-cyan-400/50"
              />
            </label>
            <label className="text-xs text-slate-400">
              Minimum consecutive idle frames
              <input
                type="number"
                min={1}
                value={minimumRun}
                disabled={running}
                onChange={(event) =>
                  setMinimumRun(Math.max(1, Number(event.target.value) || 1))
                }
                className="mt-2 block w-full rounded border border-white/10 bg-black/20 px-3 py-2 font-mono text-sm text-slate-200 outline-none focus:border-cyan-400/50"
              />
            </label>
          </section>

          <section className="panel p-5">
            <h2 className="text-sm font-medium text-slate-200">Idle match</h2>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              {[
                [
                  "both",
                  "Both arms idle",
                  "Recommended: remove only when left and right are idle.",
                ],
                [
                  "left",
                  "Left arm idle",
                  "Remove when left is idle, even if right is moving.",
                ],
                [
                  "right",
                  "Right arm idle",
                  "Remove when right is idle, even if left is moving.",
                ],
                [
                  "either",
                  "Left or right idle",
                  "Aggressive: remove whenever either arm is idle.",
                ],
              ].map(([value, label, description]) => (
                <label
                  key={value}
                  className={`cursor-pointer rounded border p-3 ${
                    idleMode === value
                      ? "border-cyan-400/40 bg-cyan-400/5"
                      : "border-white/5 bg-black/10"
                  }`}
                >
                  <span className="flex items-center gap-2 text-xs text-slate-200">
                    <input
                      type="radio"
                      name="idleMode"
                      value={value}
                      checked={idleMode === value}
                      disabled={running}
                      onChange={(event) => setIdleMode(event.target.value)}
                    />
                    {label}
                  </span>
                  <span className="mt-1 block pl-5 text-[10px] leading-relaxed text-slate-500">
                    {description}
                  </span>
                </label>
              ))}
            </div>
          </section>

          <section className="panel p-5">
            <h2 className="text-sm font-medium text-slate-200">
              Episode outcomes to modify
            </h2>
            <div className="mt-3 flex flex-wrap gap-4 text-xs text-slate-300">
              {[
                ["both", "Success and failure"],
                ["success", "Success only"],
                ["failure", "Failure only"],
              ].map(([value, label]) => (
                <label key={value} className="flex items-center gap-2">
                  <input
                    type="radio"
                    name="outcomeScope"
                    value={value}
                    checked={outcomeScope === value}
                    disabled={running}
                    onChange={(event) => setOutcomeScope(event.target.value)}
                  />
                  {label}
                </label>
              ))}
            </div>
            {outcomeScope !== "both" && (
              <p className="mt-3 text-[10px] text-slate-500">
                Other outcomes are copied without removing frames.
              </p>
            )}
          </section>

          {idleMode === "either" && (
            <div className="rounded border border-amber-400/20 bg-amber-400/5 p-3 text-xs text-amber-200">
              “Left or right” can remove useful one-arm behavior whenever the
              other arm is stationary. Inspect this rule carefully.
            </div>
          )}
          {error && (
            <div className="rounded border border-red-400/30 bg-red-400/10 p-3 text-sm text-red-200">
              {error}
            </div>
          )}

          {status && (
            <section
              className={`rounded border p-4 ${
                status.state === "failed"
                  ? "border-red-400/30 bg-red-400/5"
                  : "border-cyan-400/20 bg-cyan-400/5"
              }`}
            >
              <div className="flex justify-between text-xs">
                <span>{status.message}</span>
                <span className="tabular text-slate-400">
                  {progress.toFixed(1)}%
                </span>
              </div>
              <div className="mt-3 h-2 overflow-hidden rounded bg-black/30">
                <div
                  className="h-full bg-cyan-400 transition-all"
                  style={{ width: `${progress}%` }}
                />
              </div>
              {(status.keptFrames !== undefined ||
                status.removedFrames !== undefined) && (
                <p className="mt-3 text-[11px] text-slate-400">
                  Episode {status.currentEpisode || 0}/
                  {status.totalEpisodes || "—"}
                  {status.currentEpisodeFrames
                    ? ` · frame ${(status.currentFrame || 0).toLocaleString()}/${status.currentEpisodeFrames.toLocaleString()}`
                    : ""}
                  {" · "}
                  Kept {(status.keptFrames || 0).toLocaleString()} · removed{" "}
                  {(status.removedFrames || 0).toLocaleString()} frames
                </p>
              )}
              {status.state === "complete" && (
                <Link
                  href={`/local/${encodeURIComponent(status.output)}/episode_0`}
                  className="mt-4 inline-block rounded bg-cyan-400/10 px-3 py-2 text-xs text-cyan-300 hover:bg-cyan-400/20"
                >
                  Open {status.output}
                </Link>
              )}
            </section>
          )}

          <button
            type="submit"
            disabled={running || !source || !output}
            className="rounded bg-cyan-400/15 px-5 py-2.5 text-sm font-medium text-cyan-200 ring-1 ring-cyan-400/30 hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? "Filtering dataset…" : "Create filtered dataset"}
          </button>
        </form>
      </div>
    </main>
  );
}
