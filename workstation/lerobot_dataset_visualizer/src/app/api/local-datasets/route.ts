import { readdir, readFile, realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { resolve } from "node:path";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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

export async function GET(): Promise<Response> {
  const root = await realpath(dataRoot()).catch(() => null);
  if (!root) {
    return Response.json(
      { root: dataRoot(), datasets: [], error: "Dataset root not found" },
      { status: 404 },
    );
  }

  const entries = await readdir(root, { withFileTypes: true });
  const datasets = await Promise.all(
    entries
      .filter((entry) => entry.isDirectory() || entry.isSymbolicLink())
      .map(async (entry) => {
        const datasetPath = resolve(root, entry.name);
        try {
          const [rawInfo, datasetStat] = await Promise.all([
            readFile(resolve(datasetPath, "meta", "info.json"), "utf8"),
            stat(datasetPath),
          ]);
          const info = JSON.parse(rawInfo) as Record<string, unknown>;
          if (!info.features || !info.codebase_version) return null;
          return {
            name: entry.name,
            path: datasetPath,
            version: String(info.codebase_version),
            robotType: info.robot_type ? String(info.robot_type) : null,
            episodes: Number(info.total_episodes || 0),
            frames: Number(info.total_frames || 0),
            fps: Number(info.fps || 0),
            modifiedMs: datasetStat.mtimeMs,
          };
        } catch {
          return null;
        }
      }),
  );

  return Response.json({
    root,
    datasets: datasets
      .filter((dataset) => dataset !== null)
      .sort((a, b) => b.modifiedMs - a.modifiedMs),
  });
}
