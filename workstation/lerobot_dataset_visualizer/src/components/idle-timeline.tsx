"use client";

import { useEffect, useMemo, useState } from "react";
import { useTime } from "@/context/time-context";
import {
  findIdleRuns,
  type ActionDeltaFrame,
  type IdleArm,
  type IdleRun,
} from "@/utils/idleAnalysis";

type Props = {
  samples: ActionDeltaFrame[];
  fps: number;
};

const TRACKS: Array<{ arm: IdleArm; label: string; color: string }> = [
  { arm: "left", label: "Left idle", color: "bg-sky-400" },
  { arm: "right", label: "Right idle", color: "bg-amber-400" },
  { arm: "both", label: "Both idle", color: "bg-rose-400" },
];

const SETTINGS_KEY = "i2rt-idle-timeline-settings";

function seconds(value: number): string {
  return `${value.toFixed(2)}s`;
}

function Track({
  label,
  color,
  runs,
  duration,
  currentTime,
  onSeek,
}: {
  label: string;
  color: string;
  runs: IdleRun[];
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}) {
  return (
    <div className="grid grid-cols-[5.5rem_1fr_4.5rem] items-center gap-3">
      <span className="text-right text-[11px] text-slate-400">{label}</span>
      <div
        className="relative h-7 cursor-crosshair overflow-hidden rounded-sm border border-white/5 bg-black/30"
        onClick={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          onSeek(((event.clientX - rect.left) / rect.width) * duration);
        }}
      >
        {runs.map((run) => {
          const left = (run.startTime / duration) * 100;
          const width = Math.max(
            ((run.endTime - run.startTime) / duration) * 100,
            0.12,
          );
          return (
            <button
              key={`${run.startFrame}-${run.endFrame}`}
              className={`absolute inset-y-1 rounded-sm ${color} opacity-80 hover:opacity-100`}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`frames ${run.startFrame}–${run.endFrame} · ${seconds(run.startTime)}–${seconds(run.endTime)} · ${run.frames} frames`}
              onClick={(event) => {
                event.stopPropagation();
                onSeek(run.startTime);
              }}
            />
          );
        })}
        <div
          className="pointer-events-none absolute inset-y-0 w-px bg-white shadow-[0_0_4px_rgba(255,255,255,0.8)]"
          style={{
            left: `${Math.min(100, Math.max(0, (currentTime / duration) * 100))}%`,
          }}
        />
      </div>
      <span className="tabular text-[10px] text-slate-500">
        {runs.length} runs
      </span>
    </div>
  );
}

export default function IdleTimeline({ samples, fps }: Props) {
  const { duration: playerDuration, currentTime, seek } = useTime();
  const [thresholdText, setThresholdText] = useState("0.001");
  const [minFrames, setMinFrames] = useState(1);
  const [settingsLoaded, setSettingsLoaded] = useState(false);

  useEffect(() => {
    try {
      const stored = window.localStorage.getItem(SETTINGS_KEY);
      if (stored) {
        const parsed = JSON.parse(stored) as {
          threshold?: unknown;
          minFrames?: unknown;
        };
        const storedThreshold = Number(parsed.threshold);
        const storedMinFrames = Number(parsed.minFrames);
        if (Number.isFinite(storedThreshold) && storedThreshold >= 0) {
          setThresholdText(String(storedThreshold));
        }
        if (Number.isInteger(storedMinFrames) && storedMinFrames >= 1) {
          setMinFrames(storedMinFrames);
        }
      }
    } catch {
      // Invalid or unavailable browser storage: retain the defaults.
    } finally {
      setSettingsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (!settingsLoaded) return;
    try {
      window.localStorage.setItem(
        SETTINGS_KEY,
        JSON.stringify({ threshold: thresholdText, minFrames }),
      );
    } catch {
      // Storage can be disabled; the controls still work for this episode.
    }
  }, [minFrames, settingsLoaded, thresholdText]);

  const threshold = Math.max(0, Number(thresholdText) || 0);
  const duration = Math.max(
    playerDuration,
    samples.at(-1)?.timestamp || 0,
    1 / Math.max(fps, 1),
  );

  const runsByArm = useMemo(() => {
    return Object.fromEntries(
      TRACKS.map(({ arm }) => [
        arm,
        findIdleRuns(samples, threshold, minFrames, arm, fps),
      ]),
    ) as Record<IdleArm, IdleRun[]>;
  }, [fps, minFrames, samples, threshold]);

  const activeSample = useMemo(() => {
    if (!samples.length) return null;
    let best = samples[0];
    for (const sample of samples) {
      if (
        Math.abs(sample.timestamp - currentTime) <
        Math.abs(best.timestamp - currentTime)
      )
        best = sample;
    }
    return best;
  }, [currentTime, samples]);

  const bothFrames = runsByArm.both.reduce((sum, run) => sum + run.frames, 0);

  if (!samples.length) {
    return (
      <div className="panel p-4 text-sm text-slate-500">
        Idle timeline unavailable: this episode does not contain a 14-D action
        column.
      </div>
    );
  }

  return (
    <section className="panel p-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-sm font-medium text-slate-200">
            Per-arm idle transitions
          </h2>
          <p className="mt-1 text-[11px] text-slate-500">
            Idle when max |action[t] − action[t−1]| is strictly below the
            threshold. Click a band to seek.
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-[10px] uppercase tracking-wide text-slate-500">
            Threshold
            <input
              value={thresholdText}
              onChange={(event) => setThresholdText(event.target.value)}
              inputMode="decimal"
              className="mt-1 block w-28 rounded border border-white/10 bg-black/20 px-2 py-1.5 font-mono text-xs normal-case text-slate-200 outline-none focus:border-cyan-400/50"
            />
          </label>
          <label className="text-[10px] uppercase tracking-wide text-slate-500">
            Minimum run
            <input
              type="number"
              min={1}
              step={1}
              value={minFrames}
              onChange={(event) =>
                setMinFrames(Math.max(1, Number(event.target.value) || 1))
              }
              className="mt-1 block w-24 rounded border border-white/10 bg-black/20 px-2 py-1.5 font-mono text-xs normal-case text-slate-200 outline-none focus:border-cyan-400/50"
            />
          </label>
          <button
            onClick={() => {
              setThresholdText("0.001");
              setMinFrames(7);
            }}
            className="rounded border border-cyan-400/20 bg-cyan-400/5 px-2.5 py-1.5 text-[11px] text-cyan-300 hover:bg-cyan-400/10"
          >
            OpenPI preset
          </button>
        </div>
      </div>

      <div className="mt-5 space-y-2">
        {TRACKS.map((track) => (
          <Track
            key={track.arm}
            {...track}
            runs={runsByArm[track.arm]}
            duration={duration}
            currentTime={currentTime}
            onSeek={seek}
          />
        ))}
      </div>

      <div className="mt-4 flex flex-wrap gap-x-6 gap-y-2 border-t border-white/5 pt-3 text-[11px] text-slate-500">
        <span>
          Both-idle frames in displayed runs:{" "}
          <strong className="tabular font-medium text-rose-300">
            {bothFrames}
          </strong>
        </span>
        <span>
          Current frame:{" "}
          <strong className="tabular font-medium text-slate-300">
            {activeSample?.frameIndex ?? "—"}
          </strong>
        </span>
        <span>
          left Δ{" "}
          <strong className="font-mono font-medium text-sky-300">
            {activeSample?.leftDelta.toExponential(3) ?? "—"}
          </strong>
        </span>
        <span>
          right Δ{" "}
          <strong className="font-mono font-medium text-amber-300">
            {activeSample?.rightDelta.toExponential(3) ?? "—"}
          </strong>
        </span>
      </div>

      {runsByArm.both.length > 0 && (
        <div className="mt-3 max-h-32 overflow-y-auto rounded border border-white/5 bg-black/15">
          {runsByArm.both.map((run) => (
            <button
              key={`${run.startFrame}-${run.endFrame}`}
              onClick={() => seek(run.startTime)}
              className="grid w-full grid-cols-[1fr_1fr_1fr] gap-3 border-b border-white/5 px-3 py-1.5 text-left font-mono text-[10px] text-slate-500 last:border-0 hover:bg-white/5 hover:text-slate-300"
            >
              <span>
                frames {run.startFrame}–{run.endFrame}
              </span>
              <span>
                {seconds(run.startTime)}–{seconds(run.endTime)}
              </span>
              <span>{run.frames} frames</span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
