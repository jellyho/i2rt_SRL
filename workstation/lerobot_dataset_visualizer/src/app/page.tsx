"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

type LocalDataset = {
  name: string;
  path: string;
  version: string;
  robotType: string | null;
  episodes: number;
  frames: number;
  fps: number;
  modifiedMs: number;
};

type DatasetResponse = {
  root: string;
  datasets: LocalDataset[];
  error?: string;
};

export default function Home() {
  const [result, setResult] = useState<DatasetResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");

  const refresh = async () => {
    setError(null);
    try {
      const response = await fetch("/api/local-datasets", {
        cache: "no-store",
      });
      const body = (await response.json()) as DatasetResponse;
      if (!response.ok)
        throw new Error(body.error || `HTTP ${response.status}`);
      setResult(body);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  const datasets = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return result?.datasets || [];
    return (result?.datasets || []).filter((dataset) =>
      `${dataset.name} ${dataset.robotType || ""}`
        .toLowerCase()
        .includes(needle),
    );
  }, [query, result]);

  return (
    <main className="min-h-screen bg-[var(--bg)] px-6 py-10 text-slate-100">
      <div className="mx-auto max-w-6xl">
        <div className="flex flex-col gap-5 border-b border-white/10 pb-7 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-medium uppercase tracking-[0.22em] text-cyan-400">
              i2rt · offline
            </p>
            <h1 className="mt-2 text-3xl font-semibold tracking-tight">
              LeRobot Dataset Visualizer
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Discovering datasets under{" "}
              <span className="font-mono text-slate-300">
                {result?.root || "~/lerobot_data"}
              </span>
            </p>
          </div>
          <div className="flex gap-2">
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Filter datasets"
              className="w-64 rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm outline-none placeholder:text-slate-600 focus:border-cyan-400/60"
            />
            <button
              onClick={() => void refresh()}
              className="rounded-md border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-300 hover:border-cyan-400/40 hover:text-cyan-300"
            >
              Refresh
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-6 rounded-md border border-red-400/30 bg-red-400/10 p-4 text-sm text-red-200">
            {error}. Set <span className="font-mono">LEROBOT_DATA_ROOT</span>{" "}
            when starting the app to use another directory.
          </div>
        )}

        {!result && !error && (
          <p className="mt-10 text-sm text-slate-500">
            Scanning local datasets…
          </p>
        )}

        {result && datasets.length === 0 && (
          <div className="mt-10 rounded-md border border-dashed border-white/10 p-10 text-center text-sm text-slate-500">
            No LeRobot datasets with{" "}
            <span className="font-mono">meta/info.json</span> found.
          </div>
        )}

        <div className="mt-7 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {datasets.map((dataset) => (
            <div
              key={dataset.name}
              className="group rounded-lg border border-white/10 bg-white/[0.025] p-5 transition hover:-translate-y-0.5 hover:border-cyan-400/40 hover:bg-cyan-400/[0.04]"
            >
              <div className="flex items-start justify-between gap-3">
                <h2 className="truncate text-lg font-medium text-slate-100 group-hover:text-cyan-300">
                  {dataset.name}
                </h2>
                <span className="rounded bg-white/5 px-2 py-1 font-mono text-[10px] text-slate-500">
                  {dataset.version}
                </span>
              </div>
              <p
                className="mt-1 truncate font-mono text-[11px] text-slate-600"
                title={dataset.path}
              >
                {dataset.path}
              </p>
              <dl className="mt-5 grid grid-cols-3 gap-3 border-t border-white/5 pt-4 text-xs">
                <div>
                  <dt className="text-slate-600">Episodes</dt>
                  <dd className="mt-1 tabular text-slate-300">
                    {dataset.episodes.toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-600">Frames</dt>
                  <dd className="mt-1 tabular text-slate-300">
                    {dataset.frames.toLocaleString()}
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-600">Rate</dt>
                  <dd className="mt-1 tabular text-slate-300">
                    {dataset.fps} Hz
                  </dd>
                </div>
              </dl>
              {dataset.robotType && (
                <p className="mt-4 text-[11px] text-slate-500">
                  {dataset.robotType}
                </p>
              )}
              <div className="mt-4 flex gap-2 border-t border-white/5 pt-4">
                <Link
                  href={`/local/${encodeURIComponent(dataset.name)}/episode_0`}
                  className="rounded bg-cyan-400/10 px-3 py-1.5 text-xs text-cyan-300 hover:bg-cyan-400/20"
                >
                  Visualize
                </Link>
                <Link
                  href={`/filter?dataset=${encodeURIComponent(dataset.name)}`}
                  className="rounded border border-white/10 px-3 py-1.5 text-xs text-slate-400 hover:border-cyan-400/30 hover:text-cyan-300"
                >
                  Filter copy
                </Link>
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
