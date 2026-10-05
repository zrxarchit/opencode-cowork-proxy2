/**
 * Resolves the latest stable OpenCode release version via the GitHub API.
 * Used to build the official-client User-Agent (`opencode/stable/<v>/opencode`).
 *
 * - Endpoint: GET https://api.github.com/repos/anomalyco/opencode/releases/latest
 *   (`/latest` already excludes drafts and prereleases, i.e. stable only.)
 * - Auth is optional: unauthenticated calls get 60 req/hour (shared across all
 *   users behind the same egress IP), a token raises it to 5,000 req/hour.
 *   Provide it as the `GITHUB_API_KEY` Worker secret
 *   (`npx wrangler secret put GITHUB_API_KEY`); never commit it.
 * - The version is cached in-memory for VERSION_TTL_MS. Releases ship rarely,
 *   so re-checking GitHub on every proxied chat request would only add latency.
 *   Set VERSION_TTL_MS to 0 to check GitHub on every request instead.
 * - Any failure (no network, rate limit, bad payload, no secret) falls back to
 *   DEFAULT_OPENCODE_VERSION, so chat requests never break because of this.
 */

const RELEASES_URL = "https://api.github.com/repos/anomalyco/opencode/releases/latest";

export const DEFAULT_OPENCODE_VERSION = "2.0.23";
// 5-minute in-memory cache: survives across warm isolates on the Cloudflare
// edge, so GitHub is hit at most ~12x/hour per isolate instead of per request.
export const VERSION_TTL_MS = 5 * 60 * 1000;

let cached: { version: string; at: number } | null = null;

/** Test-only hook to clear the in-memory version cache. */
export function resetVersionCache(): void {
  cached = null;
}

export async function getOpenCodeVersion(env?: { GITHUB_API_KEY?: string }): Promise<string> {
  const now = Date.now();
  if (cached && now - cached.at < VERSION_TTL_MS) return cached.version;
  try {
    const headers: Record<string, string> = {
      Accept: "application/vnd.github+json",
      // GitHub API rejects requests without a User-Agent.
      "User-Agent": "opencode-proxy",
    };
    const token = env?.GITHUB_API_KEY;
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await fetch(RELEASES_URL, { headers });
    if (!res.ok) throw new Error(`GitHub API responded ${res.status}`);
    const data: any = await res.json();
    const tag = typeof data?.tag_name === "string" ? data.tag_name.replace(/^v/, "") : "";
    if (!/^\d+\.\d+\.\d+/.test(tag)) throw new Error(`unexpected tag_name: ${tag}`);
    cached = { version: tag, at: now };
    return tag;
  } catch {
    if (cached) return cached.version;
    return DEFAULT_OPENCODE_VERSION;
  }
}

/** Official client format, see App.useragent in opencode's packages/core/src/app.ts. */
export function openCodeUserAgent(version: string): string {
  return `opencode/stable/${version}/opencode`;
}
