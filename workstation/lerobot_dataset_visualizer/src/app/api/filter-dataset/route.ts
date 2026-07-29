import { randomUUID } from "node:crypto";
import { spawn } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync } from "node:fs";
import { readFile, realpath, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve, sep } from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const IDLE_MODES = new Set(["left", "right", "either", "both"]);
const OUTCOME_SCOPES = new Set(["both", "success", "failure"]);
const SAFE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SAFE_JOB_ID = /^[0-9a-f-]{36}$/;

function dataRoot(): string {
  const configured = process.env.LEROBOT_DATA_ROOT || "~/lerobot_data";
  return resolve(
    configured === "~"
      ? homedir()
      : configured.startsWith("~/")
        ? resolve(homedir(), configured.slice(2))
        : configured,
  );
}

function jobsRoot(): string {
  return resolve(dataRoot(), ".i2rt_filter_jobs");
}

function pythonExecutable(): string {
  const configured = process.env.I2RT_PYTHON;
  if (configured) return configured;
  const workspacePython = resolve(
    process.cwd(),
    "..",
    "..",
    ".venv",
    "bin",
    "python",
  );
  return existsSync(workspacePython) ? workspacePython : "python3";
}

export async function POST(request: Request): Promise<Response> {
  const body = (await request.json()) as Record<string, unknown>;
  const source = String(body.source || "");
  const output = String(body.output || "");
  const idleMode = String(body.idleMode || "both");
  const outcomeScope = String(body.outcomeScope || "both");
  const threshold = Number(body.threshold);
  const minimumRun = Number(body.minimumRun);

  if (!SAFE_NAME.test(source) || !SAFE_NAME.test(output)) {
    return Response.json(
      {
        error:
          "Dataset names may contain letters, numbers, dots, dashes, and underscores",
      },
      { status: 400 },
    );
  }
  if (source === output) {
    return Response.json(
      { error: "Output name must differ from the source dataset" },
      { status: 400 },
    );
  }
  if (!IDLE_MODES.has(idleMode) || !OUTCOME_SCOPES.has(outcomeScope)) {
    return Response.json(
      { error: "Invalid idle mode or outcome scope" },
      { status: 400 },
    );
  }
  if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1) {
    return Response.json(
      { error: "Threshold must be greater than 0 and at most 1" },
      { status: 400 },
    );
  }
  if (!Number.isInteger(minimumRun) || minimumRun < 1 || minimumRun > 10000) {
    return Response.json(
      { error: "Minimum run must be an integer from 1 to 10000" },
      { status: 400 },
    );
  }

  const root = await realpath(dataRoot()).catch(() => null);
  if (!root)
    return Response.json(
      { error: `Dataset root not found: ${dataRoot()}` },
      { status: 404 },
    );
  const sourceRoot = await realpath(resolve(root, source)).catch(() => null);
  if (!sourceRoot || !sourceRoot.startsWith(`${root}${sep}`)) {
    return Response.json(
      { error: `Source dataset not found: ${source}` },
      { status: 404 },
    );
  }
  if (existsSync(resolve(root, output))) {
    return Response.json(
      { error: `Output dataset already exists: ${output}` },
      { status: 409 },
    );
  }

  const jobId = randomUUID();
  mkdirSync(jobsRoot(), { recursive: true });
  const configPath = resolve(jobsRoot(), `${jobId}.config.json`);
  const statusPath = resolve(jobsRoot(), `${jobId}.status.json`);
  const logPath = resolve(jobsRoot(), `${jobId}.log`);
  const config = {
    jobId,
    dataRoot: root,
    source,
    output,
    idleMode,
    outcomeScope,
    threshold,
    minimumRun,
    maxEpisodeRemovalFraction: 0.9,
  };
  const initialStatus = {
    jobId,
    state: "queued",
    phase: "queued",
    message: "Waiting for filter process",
    source,
    output,
    createdAt: Date.now() / 1000,
    updatedAt: Date.now() / 1000,
  };
  await Promise.all([
    writeFile(configPath, JSON.stringify(config, null, 2)),
    writeFile(statusPath, JSON.stringify(initialStatus, null, 2)),
  ]);

  const script = resolve(process.cwd(), "scripts", "filter_dataset.py");
  const logFd = openSync(logPath, "a");
  const child = spawn(
    pythonExecutable(),
    [script, "--config", configPath, "--status", statusPath],
    {
      cwd: resolve(process.cwd(), "..", ".."),
      detached: true,
      stdio: ["ignore", logFd, logFd],
    },
  );
  child.unref();
  closeSync(logFd);

  return Response.json({ jobId, status: initialStatus }, { status: 202 });
}

export async function GET(request: Request): Promise<Response> {
  const jobId = new URL(request.url).searchParams.get("job_id") || "";
  if (!SAFE_JOB_ID.test(jobId))
    return Response.json({ error: "Invalid job ID" }, { status: 400 });
  const path = resolve(jobsRoot(), `${jobId}.status.json`);
  try {
    return Response.json(JSON.parse(await readFile(path, "utf8")));
  } catch {
    return Response.json({ error: "Filter job not found" }, { status: 404 });
  }
}
