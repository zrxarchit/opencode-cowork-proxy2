/**
 * Failover across free chat-protocol models.
 *
 * Free models fail often and independently: per-model rate limits (429),
 * overloaded or retired providers (503 / "Model is unavailable"). Instead of
 * surfacing the first failure, the proxy retries the same request against the
 * next free model and only gives up when every candidate fails. The client
 * always sees the model name it asked for; when failover kicked in the
 * response carries an `x-proxy-upstream-model` header naming the real model
 * that answered.
 *
 * Rules:
 * - Only free (`*-free`) chat-protocol models fail over, and never when the
 *   caller pinned a model via URL model-override (explicit choice is honored).
 * - Only retryable failures fail over: 429 / 502 / 503 / 529, 500, and 400s
 *   whose body reports an unavailable model/endpoint. Auth (401), the
 *   free-tier gate (403), and other client errors surface immediately.
 * - Failover happens before any streaming starts, so streams never switch
 *   models mid-response.
 */

import { ALIAS_TO_MODEL, isFreeModelId, RESPONSES_PROTOCOL_MODELS } from "./models";

/** Free chat-protocol models in alias-map order, used as failover candidates. */
export function chatFallbackModels(): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const real of Object.values(ALIAS_TO_MODEL)) {
    if (!isFreeModelId(real)) continue;
    if (RESPONSES_PROTOCOL_MODELS.has(real)) continue;
    if (seen.has(real)) continue;
    seen.add(real);
    out.push(real);
  }
  return out;
}

/**
 * Ordered upstream models to try: the requested model first, then every other
 * free chat model. Always returns at least the requested model, so callers can
 * loop uniformly; the loop length is 1 (no failover) unless the requested
 * model is a free chat model and the caller did not pin it via URL override.
 */
export function failoverCandidates(resolvedModel: string, hasModelOverride: boolean): string[] {
  if (!isFreeModelId(resolvedModel)) return [resolvedModel];
  if (RESPONSES_PROTOCOL_MODELS.has(resolvedModel)) return [resolvedModel];
  if (hasModelOverride) return [resolvedModel];
  const rest = chatFallbackModels().filter((m) => m !== resolvedModel);
  return [resolvedModel, ...rest];
}

/** True when an upstream failure is worth retrying against another model. */
export function isRetryableUpstream(status: number, bodyText: string): boolean {
  if (status === 429 || status === 502 || status === 503 || status === 529 || status === 500) return true;
  if (status === 400 && /unavailable/i.test(bodyText || "")) return true;
  return false;
}
