import { Hono } from 'hono';
import { extractApiKey, validateApiKey, authErrorResponse } from './auth';
import {
  resolveUpstreamModel,
  buildOpenAIModelsList,
  buildAnthropicModelsList,
  buildMapResponse,
} from './models';
import { formatAnthropicToOpenAI } from './translate/request/anthropic-to-openai';
import { formatOpenAIToAnthropic } from './translate/request/openai-to-anthropic';
import { formatOpenAIToAnthropic as toAnthropicResponse } from './translate/response/openai-to-anthropic';
import { formatAnthropicToOpenAI as toOpenAIResponse } from './translate/response/anthropic-to-openai';
import { streamOpenAIToAnthropic } from './translate/stream/openai-to-anthropic';
import { streamAnthropicToOpenAI } from './translate/stream/anthropic-to-openai';

const GO_UPSTREAM = "https://opencode.ai/zen/go/v1";
const ZEN_UPSTREAM = "https://opencode.ai/zen/v1";
const DEFAULT_UPSTREAM = GO_UPSTREAM;
const VISION_MODEL = "qwen3.6-plus";

const API_START_PATHS = new Set(['v1', 'v2']);
const RESERVED_SEGMENTS = new Set(['map']);

type RouteConfig = {
  path: string;
  upstream: string;
  modelOverride: string | null;
};

function stripPrefix(path: string, prefix: string): string | null {
  if (path === prefix) return "/";
  if (path.startsWith(`${prefix}/`)) return path.slice(prefix.length);
  return null;
}

function extractModelSegment(path: string): { path: string; model: string | null } {
  const segments = path.replace(/^\/+/, '').split('/');
  if (segments.length > 0 && segments[0] && !API_START_PATHS.has(segments[0]) && !RESERVED_SEGMENTS.has(segments[0])) {
    return { path: '/' + segments.slice(1).join('/'), model: segments[0] };
  }
  return { path, model: null };
}

function routeConfig(request: Request): RouteConfig {
  const path = new URL(request.url).pathname;
  const goPath = stripPrefix(path, "/go");
  if (goPath) {
    const { path: remaining, model } = extractModelSegment(goPath);
    return { path: remaining, upstream: GO_UPSTREAM, modelOverride: model };
  }

  const zenPath = stripPrefix(path, "/zen");
  if (zenPath) {
    const { path: remaining, model } = extractModelSegment(zenPath);
    return { path: remaining, upstream: ZEN_UPSTREAM, modelOverride: model };
  }

  const { path: remaining, model } = extractModelSegment(path);
  return { path: remaining, upstream: DEFAULT_UPSTREAM, modelOverride: model };
}

function getUpstream(request: Request, routeUpstream: string): string {
  return request.headers.get("X-Upstream-Url") || routeUpstream;
}

function upstreamFormat(request: Request): "openai" | "anthropic" {
  const fmt = (request.headers.get("X-Upstream-Format") || "openai").toLowerCase();
  return fmt === "anthropic" ? "anthropic" : "openai";
}

function anthropicHeaders(request: Request, key: string): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Api-Key": key,
    "Anthropic-Version": request.headers.get("Anthropic-Version") || "2023-06-01",
  };
  const beta = request.headers.get("Anthropic-Beta");
  if (beta) headers["Anthropic-Beta"] = beta;
  return headers;
}

function hasImages(body: any): boolean {
  const messages = body?.messages;
  if (!Array.isArray(messages)) return false;
  return messages.some((msg: any) =>
    Array.isArray(msg.content) && msg.content.some((part: any) => part.type === "image")
  );
}

function upstreamErrorResponse(res: Response, body: string): Response {
  const headers = new Headers();
  for (const name of ["Content-Type", "Retry-After", "RateLimit-Limit", "RateLimit-Remaining", "RateLimit-Reset"]) {
    const value = res.headers.get(name);
    if (value) headers.set(name, value);
  }
  return new Response(body, { status: res.status, headers });
}

async function handleRequest(request: Request): Promise<Response> {
  const route = routeConfig(request);
  const upstream = getUpstream(request, route.upstream);
  const fmt = upstreamFormat(request);

  // Anthropic → OpenAI (for Claude Desktop/Cowork → any OpenAI API)
  if (route.path === '/v1/messages' && request.method === 'POST') {
      const key = extractApiKey(request.headers);
      const err = validateApiKey(key);
      if (err) return authErrorResponse(err);

      if (fmt === "openai") {
        const req: any = await request.json();
        const originalModel = req.model;
        if (route.modelOverride) req.model = route.modelOverride;
        req.model = resolveUpstreamModel(req.model) ?? req.model;
        if (hasImages(req)) {
          req.model = VISION_MODEL;
        }
        const openaiReq = formatAnthropicToOpenAI(req);
        const res = await fetch(`${upstream}/chat/completions`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Bearer ${key}`,
          },
          body: JSON.stringify(openaiReq),
        });
        if (!res.ok) return upstreamErrorResponse(res, await res.text());

        if (openaiReq.stream) {
          return new Response(streamOpenAIToAnthropic(res.body as ReadableStream, originalModel), {
            headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" },
          });
        }
        const data: any = await res.json();
        return new Response(JSON.stringify(toAnthropicResponse(data, originalModel)), {
          headers: { "Content-Type": "application/json" },
        });
      }

      // Pass-through to Anthropic upstream (still resolves aliases / URL override)
      const rawText = await request.text();
      let passthroughBody = rawText;
      try {
        const parsed: any = JSON.parse(rawText);
        if (route.modelOverride) parsed.model = route.modelOverride;
        parsed.model = resolveUpstreamModel(parsed.model) ?? parsed.model;
        passthroughBody = JSON.stringify(parsed);
      } catch {
        // non-JSON body: forward as-is
      }
      const res = await fetch(`${upstream}/v1/messages`, {
        method: "POST",
        headers: anthropicHeaders(request, key!),
        body: passthroughBody,
      });
      return res;
  }

  // OpenAI → Anthropic (or pass-through)
  if (route.path === '/v1/chat/completions' && request.method === 'POST') {
      const key = extractApiKey(request.headers);
      const err = validateApiKey(key);
      if (err) return authErrorResponse(err);

      if (fmt === "anthropic") {
        const req: any = await request.json();
        const originalModel = req.model;
        if (route.modelOverride) req.model = route.modelOverride;
        req.model = resolveUpstreamModel(req.model) ?? req.model;
        const anthReq = formatOpenAIToAnthropic(req);
        const res = await fetch(`${upstream}/v1/messages`, {
          method: "POST",
          headers: anthropicHeaders(request, key!),
          body: JSON.stringify(anthReq),
        });
        if (!res.ok) return upstreamErrorResponse(res, await res.text());

        if (anthReq.stream) {
          return new Response(streamAnthropicToOpenAI(res.body as ReadableStream, originalModel), {
            headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive" },
          });
        }
        const data: any = await res.json();
        return new Response(JSON.stringify(toOpenAIResponse(data, originalModel)), {
          headers: { "Content-Type": "application/json" },
        });
      }

      // Pass-through to OpenAI upstream (still resolves aliases / URL override)
      const rawOpenAIText = await request.text();
      let openAIBody = rawOpenAIText;
      let openAIOriginalModel: string | null = null;
      let openAIResolvedModel: string | null = null;
      try {
        const parsed: any = JSON.parse(rawOpenAIText);
        openAIOriginalModel = parsed.model ?? null;
        if (route.modelOverride) parsed.model = route.modelOverride;
        parsed.model = resolveUpstreamModel(parsed.model) ?? parsed.model;
        openAIResolvedModel = parsed.model ?? null;
        openAIBody = JSON.stringify(parsed);
      } catch {
        // non-JSON body: forward as-is
      }
      const res = await fetch(`${upstream}/chat/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "Authorization": `Bearer ${key}` },
        body: openAIBody,
      });
      if (res.ok && openAIOriginalModel && openAIResolvedModel && openAIOriginalModel !== openAIResolvedModel) {
        // Map the response model back to the alias the client asked for.
        try {
          const data: any = await res.json();
          if (data && typeof data.model === "string") data.model = openAIOriginalModel;
          return new Response(JSON.stringify(data), {
            headers: { "Content-Type": "application/json" },
          });
        } catch {
          return res;
        }
      }
      return res;
  }

  // Public alias map: which alias resolves to which real upstream model.
  // Works with /v1/map, /go/v1/map, /zen/v1/map — no API key required, no upstream call.
  if (route.path === '/v1/map' && request.method === 'GET') {
      return new Response(JSON.stringify(buildMapResponse(), null, 2), {
        headers: { "Content-Type": "application/json" },
      });
  }

  // Public model discovery: lists ONLY free-model aliases
  // (e.g. claude-opus-x-1, claude-opus-x-2, ...) — no API key required,
  // no upstream call. Real upstream IDs are never exposed here.
  if (route.path === '/v1/models' && request.method === 'GET') {
      const body = fmt === "anthropic" ? buildAnthropicModelsList() : buildOpenAIModelsList();
      return new Response(JSON.stringify(body), { headers: { "Content-Type": "application/json" } });
  }

  return new Response(JSON.stringify({
    name: "opencode-cowork-proxy",
    upstream,
    routes: {
      "/go": GO_UPSTREAM,
      "/zen": ZEN_UPSTREAM,
    },
    endpoints: {
      "/v1/messages": "Anthropic → upstream (translated if upstream=openai, aliases resolved)",
      "/v1/chat/completions": "OpenAI → upstream (translated if upstream=anthropic, aliases resolved)",
      "/v1/models": "Public free-model alias list (no auth, e.g. claude-opus-x-1...)",
      "/v1/map": "Public alias -> real upstream model map (no auth)",
    },
  }, null, 2), {
    headers: { "Content-Type": "application/json" },
    status: route.path === '/' ? 200 : 404,
  });
}

const app = new Hono();
app.all('*', (c) => handleRequest(c.req.raw));

export default app;
