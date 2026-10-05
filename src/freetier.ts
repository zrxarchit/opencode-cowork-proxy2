/**
 * OpenCode free-tier client contract (reverse-engineered, verified live).
 *
 * The server answers free (`*-free`) models with
 * `403 FreeTierError: OpenCode's free tier can only be used from within OpenCode`
 * unless the request looks like it comes from the official app. Headers alone
 * are NOT enough — the request BODY must also satisfy:
 *   1. `stream: true` (non-streaming requests are rejected outright), and
 *   2. a `tools` array declaring functions named `shell` (or `bash`) and `read`.
 *
 * This module enforces that contract on OpenAI-format upstream bodies for free
 * models only. Paid models pass through untouched.
 *
 * Caveat: the injected `shell`/`read` compatibility tools are visible to the
 * model, which may occasionally invoke them. Real OpenCode clients declare
 * these tools natively so this never bites there; other clients should treat
 * calls to unknown tools as no-ops/errors.
 */

import { isFreeModelId } from "./models";

export interface ClientTool {
  name: string;
  schema?: any;
}

function compatTool(name: string, description: string, schema: any): any {
  return {
    type: "function",
    function: {
      name,
      description,
      parameters: schema,
    },
  };
}

/**
 * Mutate an OpenAI-format request body in place to satisfy the free-tier gate.
 * Returns the names injected (empty when the model is not free).
 *
 * Injected compat tools mirror the parameter schema of the client's own
 * Bash/Read-like tools when present, so that when the model invokes them the
 * translated call fits the client's tool and can be renamed back to it.
 *
 * `toolShape` selects the function-tool envelope: chat-completions bodies nest
 * it under `function`, while the Responses API declares it flat. Sending the
 * wrong shape means the gate sees no usable tools and answers 403.
 */
export function enforceFreeTierContract(
  openaiBody: any,
  clientTools: ClientTool[] = [],
  toolShape: "chat" | "responses" = "chat",
): string[] {
  if (!openaiBody || typeof openaiBody.model !== "string" || !isFreeModelId(openaiBody.model)) {
    return [];
  }
  openaiBody.stream = true;
  if (toolShape === "chat") {
    openaiBody.stream_options = { ...(openaiBody.stream_options || {}), include_usage: true };
  }
  const tools = Array.isArray(openaiBody.tools) ? openaiBody.tools : (openaiBody.tools = []);
  const names = new Set(
    tools.map((t: any) => t?.function?.name || t?.name).filter((n: any) => typeof n === "string"),
  );
  const schemaFor = (candidates: string[]): any => {
    const hit = clientTools.find((t) => candidates.includes(t.name.toLowerCase()));
    return hit?.schema && typeof hit.schema === "object"
      ? hit.schema
      : { type: "object", properties: {} };
  };
  const inject = (name: string, description: string, schema: any) => {
    if (toolShape === "responses") {
      tools.push({ type: "function", name, description, parameters: schema });
    } else {
      tools.push(compatTool(name, description, schema));
    }
  };
  const injected: string[] = [];
  if (!names.has("shell") && !names.has("bash")) {
    inject("shell", "Run a shell command", schemaFor(["bash", "shell"]));
    injected.push("shell");
  }
  if (!names.has("read")) {
    inject("read", "Read a file", schemaFor(["read"]));
    injected.push("read");
  }
  return injected;
}

/** `bash` and `shell` are equivalent spellings of the same tool. */
function canonicalToolName(name: string): string {
  const lower = name.toLowerCase();
  return lower === "bash" ? "shell" : lower;
}

/**
 * Map injected compat-tool names back to the client's own tool names
 * (e.g. `shell` -> `Bash`) so translated tool calls are executable downstream.
 * Only maps when the client did not declare the exact injected name.
 */
export function buildToolNameMap(injected: string[], clientTools: ClientTool[]): Record<string, string> {
  const map: Record<string, string> = {};
  const clientNames = clientTools.map((t) => t.name);
  for (const name of injected) {
    if (clientNames.includes(name)) continue;
    const match = clientNames.find((n) => canonicalToolName(n) === canonicalToolName(name));
    if (match) map[name] = match;
  }
  return map;
}

/**
 * Collect a streaming OpenAI chat-completions SSE response into a single
 * standard completion object (for clients that did not ask for streaming).
 */
export async function collectOpenAIStream(res: Response): Promise<any> {
  const text = await res.text();
  let content = "";
  let reasoning = "";
  const toolCalls: Record<number, { id?: string; name?: string; args?: string }> = {};
  let finish: string | null = null;
  let usage: any = undefined;
  let id = `chatcmpl-${Date.now()}`;
  let model = "";
  const created = Math.floor(Date.now() / 1000);

  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const data = trimmed.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    let chunk: any;
    try {
      chunk = JSON.parse(data);
    } catch {
      continue;
    }
    if (typeof chunk.id === "string") id = chunk.id;
    if (typeof chunk.model === "string") model = chunk.model;
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    const delta = choice.delta || {};
    if (typeof delta.content === "string") content += delta.content;
    if (typeof delta.reasoning_content === "string") reasoning += delta.reasoning_content;
    // Some providers stream `reasoning` instead of `reasoning_content`.
    if (typeof delta.reasoning === "string") reasoning += delta.reasoning;
    for (const tc of delta.tool_calls || []) {
      const index = typeof tc.index === "number" ? tc.index : 0;
      const slot = (toolCalls[index] = toolCalls[index] || {});
      if (typeof tc.id === "string") slot.id = tc.id;
      if (typeof tc.function?.name === "string") slot.name = tc.function.name;
      if (typeof tc.function?.arguments === "string") slot.args = (slot.args || "") + tc.function.arguments;
    }
    if (typeof choice.finish_reason === "string") finish = choice.finish_reason;
  }

  const message: any = { role: "assistant", content: content || null };
  if (reasoning) message.reasoning_content = reasoning;
  const calls = Object.values(toolCalls).filter((s) => s.name || s.id);
  if (calls.length > 0) {
    message.tool_calls = calls.map((s, i) => ({
      id: s.id || `call_${i}`,
      type: "function",
      function: { name: s.name || "", arguments: s.args || "" },
    }));
    if (!finish) finish = "tool_calls";
  }

  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [{ index: 0, message, finish_reason: finish || "stop" }],
    ...(usage ? { usage } : {}),
  };
}
