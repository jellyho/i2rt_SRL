export type ActionDeltaFrame = {
  frameIndex: number;
  timestamp: number;
  leftDelta: number;
  rightDelta: number;
};

export type IdleArm = "left" | "right" | "both";

export type IdleRun = {
  startFrame: number;
  endFrame: number;
  startTime: number;
  endTime: number;
  frames: number;
};

function numericVector(value: unknown): number[] | null {
  if (Array.isArray(value)) {
    const values = value.map(Number);
    return values.every(Number.isFinite) ? values : null;
  }
  if (ArrayBuffer.isView(value)) {
    const values = Array.from(value as unknown as ArrayLike<number>, Number);
    return values.every(Number.isFinite) ? values : null;
  }
  return null;
}

function finiteNumber(value: unknown, fallback: number): number {
  const number = typeof value === "bigint" ? Number(value) : Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/** Compute the exact max per-arm action change before any chart downsampling. */
export function computeActionDeltas(
  rows: Record<string, unknown>[],
): ActionDeltaFrame[] {
  const result: ActionDeltaFrame[] = [];
  let previous: number[] | null = null;

  for (let index = 0; index < rows.length; index++) {
    const row = rows[index];
    const action = numericVector(row.action);
    if (!action || action.length < 14) {
      previous = null;
      continue;
    }

    const timestamp = finiteNumber(row.timestamp, index);
    const frameIndex = finiteNumber(row.frame_index, index);
    if (previous) {
      let leftDelta = 0;
      let rightDelta = 0;
      for (let joint = 0; joint < 7; joint++) {
        leftDelta = Math.max(
          leftDelta,
          Math.abs(action[joint] - previous[joint]),
        );
        rightDelta = Math.max(
          rightDelta,
          Math.abs(action[joint + 7] - previous[joint + 7]),
        );
      }
      result.push({ frameIndex, timestamp, leftDelta, rightDelta });
    }
    previous = action;
  }
  return result;
}

export function isIdle(
  sample: ActionDeltaFrame,
  threshold: number,
  arm: IdleArm,
): boolean {
  const left = sample.leftDelta < threshold;
  const right = sample.rightDelta < threshold;
  return arm === "left" ? left : arm === "right" ? right : left && right;
}

export function findIdleRuns(
  samples: ActionDeltaFrame[],
  threshold: number,
  minFrames: number,
  arm: IdleArm,
  fps: number,
): IdleRun[] {
  const runs: IdleRun[] = [];
  let start = -1;

  const closeRun = (endExclusive: number) => {
    if (start < 0) return;
    const frames = endExclusive - start;
    if (frames >= minFrames) {
      const first = samples[start];
      const last = samples[endExclusive - 1];
      runs.push({
        startFrame: first.frameIndex,
        endFrame: last.frameIndex,
        startTime: first.timestamp,
        endTime: last.timestamp + 1 / Math.max(fps, 1),
        frames,
      });
    }
    start = -1;
  };

  for (let index = 0; index < samples.length; index++) {
    if (isIdle(samples[index], threshold, arm)) {
      if (start < 0) start = index;
    } else {
      closeRun(index);
    }
  }
  closeRun(samples.length);
  return runs;
}
