/**
 * Static alias map for free upstream models.
 *
 * The proxy lists ONLY these aliases on GET /v1/models (no API key required),
 * e.g. `claude-opus-x-1`, `claude-opus-x-2`, ... instead of exposing the real
 * upstream IDs directly. POST /v1/messages and POST /v1/chat/completions
 * transparently resolve an alias back to its real upstream model ID.
 *
 * No persistence / KV needed — this list rarely changes. To update, just edit
 * ALIAS_TO_MODEL below.
 *
 * Rule for "free": upstream ID contains `-free` (e.g.
 * `muse-spark-1.3-contributor-free`, `mimo-v2.6-flash-free`,
 * `ling-3.1-flash-free`).
 */

export const ALIAS_TO_MODEL: Record<string, string> = {
  "claude-opus-x-1": "muse-spark-1.3-contributor-free",
  "claude-opus-x-2": "mimo-v2.6-flash-free",
  "claude-opus-x-3": "ling-3.1-flash-free",
  "claude-opus-x-4": "deepseek-v4-flash-free",
  "claude-opus-x-5": "mimo-v2.5-free",
  "claude-opus-x-6": "north-mini-code-free",
  "claude-opus-x-7": "nemotron-3-ultra-free",
  "claude-opus-x-8": "minimax-m2.5-free",
};

export const MODEL_TO_ALIAS: Record<string, string> = Object.fromEntries(
  Object.entries(ALIAS_TO_MODEL).map(([alias, real]) => [real, alias]),
);

export function isFreeModelId(id: string): boolean {
  return id.includes("-free");
}

export function isAlias(id: string | null | undefined): boolean {
  return !!id && id in ALIAS_TO_MODEL;
}

/** Resolve a client-supplied model (alias or real ID) to the real upstream ID. */
export function resolveUpstreamModel(model: string | null | undefined): string | null | undefined {
  if (!model) return model;
  return ALIAS_TO_MODEL[model] ?? model;
}

/** Resolve a real upstream model ID back to its public alias (if any). */
export function resolveAlias(model: string | null | undefined): string | null | undefined {
  if (!model) return model;
  return MODEL_TO_ALIAS[model] ?? model;
}

export function listAliases(): string[] {
  return Object.keys(ALIAS_TO_MODEL);
}

/**
 * Models that only speak the OpenAI Responses protocol on Zen
 * (`ModelProtocolUnsupported` on `/chat/completions` and `/messages`).
 * Add entries here as they are discovered; the proxy translates to/from
 * `/responses` for them transparently.
 */
export const RESPONSES_PROTOCOL_MODELS: ReadonlySet<string> = new Set([
  "muse-spark-1.3-contributor-free",
]);

/** OpenAI-compatible `GET /v1/models` body listing ONLY aliases. */
export function buildOpenAIModelsList() {
  const now = Math.floor(Date.now() / 1000);
  return {
    object: "list",
    data: listAliases().map((alias) => ({
      id: alias,
      object: "model",
      created: now,
      owned_by: "opencode-proxy",
    })),
  };
}

/** Anthropic-compatible `GET /v1/models` body listing ONLY aliases. */
export function buildAnthropicModelsList() {
  return {
    data: listAliases().map((alias) => ({
      type: "model",
      id: alias,
      display_name: alias,
    })),
    has_more: false,
  };
}

/** GET /v1/map body: alias -> real upstream model. */
export function buildMapResponse(): Record<string, string> {
  return { ...ALIAS_TO_MODEL };
}
