import { createReadStream } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { homedir } from "node:os";
import { extname, resolve, sep } from "node:path";
import { Readable } from "node:stream";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MIME_TYPES: Record<string, string> = {
  ".json": "application/json",
  ".jsonl": "application/x-ndjson",
  ".mp4": "video/mp4",
  ".parquet": "application/vnd.apache.parquet",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
};

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

function requestParts(
  parts: string[],
): { dataset: string; relative: string[] } | null {
  // The existing viewer requests Hub-shaped URLs:
  // /{org}/{dataset}/resolve/main/{dataset-relative path}.
  const marker = parts.findIndex(
    (part, index) => part === "resolve" && parts[index + 1] === "main",
  );
  if (marker < 2) return null;
  const dataset = decodeURIComponent(parts[marker - 1]);
  if (
    !dataset ||
    dataset === "." ||
    dataset === ".." ||
    dataset.includes("/") ||
    dataset.includes("\\")
  ) {
    return null;
  }
  return { dataset, relative: parts.slice(marker + 2).map(decodeURIComponent) };
}

async function serve(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
  headOnly = false,
): Promise<Response> {
  const parts = (await context.params).path;
  const parsed = requestParts(parts);
  if (!parsed?.relative.length) {
    return Response.json(
      {
        error: "Expected /local/<dataset>/resolve/main/<path>",
      },
      { status: 404 },
    );
  }

  const base = await realpath(dataRoot()).catch(() => null);
  if (!base) {
    return Response.json(
      { error: `Dataset root not found: ${dataRoot()}` },
      { status: 404 },
    );
  }

  const root = await realpath(resolve(base, parsed.dataset)).catch(() => null);
  if (!root || !root.startsWith(`${base}${sep}`)) {
    return Response.json(
      { error: `Dataset not found: ${parsed.dataset}` },
      { status: 404 },
    );
  }

  const candidate = resolve(root, ...parsed.relative);
  if (candidate !== root && !candidate.startsWith(`${root}${sep}`)) {
    return Response.json({ error: "Invalid dataset path" }, { status: 400 });
  }

  const file = await realpath(candidate).catch(() => null);
  if (!file || (file !== root && !file.startsWith(`${root}${sep}`))) {
    return Response.json({ error: "File not found" }, { status: 404 });
  }

  const info = await stat(file).catch(() => null);
  if (!info?.isFile()) {
    return Response.json({ error: "File not found" }, { status: 404 });
  }

  const headers = new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "no-store",
    "Content-Type":
      MIME_TYPES[extname(file).toLowerCase()] || "application/octet-stream",
  });
  const range = request.headers.get("range");
  let start = 0;
  let end = info.size - 1;
  let status = 200;

  if (range) {
    const match = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
    if (!match) {
      return new Response(null, {
        status: 416,
        headers: { "Content-Range": `bytes */${info.size}` },
      });
    }
    start = match[1] ? Number(match[1]) : 0;
    end = match[2] ? Number(match[2]) : info.size - 1;
    if (start > end || end >= info.size) {
      return new Response(null, {
        status: 416,
        headers: { "Content-Range": `bytes */${info.size}` },
      });
    }
    status = 206;
    headers.set("Content-Range", `bytes ${start}-${end}/${info.size}`);
  }

  headers.set("Content-Length", String(end - start + 1));
  if (headOnly) return new Response(null, { status, headers });

  const body = Readable.toWeb(createReadStream(file, { start, end }));
  return new Response(body as unknown as ReadableStream, { status, headers });
}

export function GET(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  return serve(request, context);
}

export function HEAD(
  request: Request,
  context: { params: Promise<{ path: string[] }> },
) {
  return serve(request, context, true);
}
