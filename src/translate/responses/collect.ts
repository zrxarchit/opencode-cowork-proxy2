/**
 * Collects a streaming OpenAI Responses API SSE body.
 *
 * Preferring the server-assembled `response.completed` event keeps full
 * fidelity (exact output items, usage) without manual delta bookkeeping.
 * Falls back to delta accumulation when the completed event is absent.
 */

interface CollectedCall {
  id?: string;
  name?: string;
  args?: string;
}

export async function collectResponsesStream(res: Response): Promise<any> {
  const text = await res.text();

  let completed: any = null;
  let createdId = `resp_${Date.now()}`;
  let model = "";
  let usage: any = undefined;
  let textOut = "";
  let reasoningOut = "";
  const calls: Record<string, CollectedCall> = {};
  const textOrder: string[] = [];

  const noteTextBlock = (key: string) => {
    if (!textOrder.includes(key)) textOrder.push(key);
  };

  for (const line of text.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) continue;
    const data = trimmed.slice(5).trim();
    if (!data || data === "[DONE]") continue;
    let event: any;
    try {
      event = JSON.parse(data);
    } catch {
      continue;
    }
    const type = event.type;
    if (type === "response.created") {
      if (typeof event.response?.id === "string") createdId = event.response.id;
      if (typeof event.response?.model === "string") model = event.response.model;
    } else if (type === "response.completed" || type === "response.incomplete") {
      if (event.response && typeof event.response === "object") {
        completed = event.response;
        if (typeof completed.id === "string") createdId = completed.id;
        if (typeof completed.model === "string") model = completed.model;
        if (completed.usage) usage = completed.usage;
      }
    } else if (type === "response.output_text.delta") {
      const key = `t${event.output_index ?? 0}`;
      noteTextBlock(key);
      if (typeof event.delta === "string") textOut += event.delta;
    } else if (type === "response.reasoning_summary_text.delta") {
      if (typeof event.delta === "string") reasoningOut += event.delta;
    } else if (type === "response.output_item.added") {
      const item = event.item || {};
      if (item.type === "function_call") {
        const key = item.call_id || item.id || `c${Object.keys(calls).length}`;
        calls[key] = { id: item.call_id || item.id, name: item.name, args: "" };
      }
    } else if (type === "response.function_call_arguments.delta") {
      const key = event.item_id || `c${Object.keys(calls).length}`;
      const slot = (calls[key] = calls[key] || {});
      if (typeof event.delta === "string") slot.args = (slot.args || "") + event.delta;
    }
  }

  if (completed && Array.isArray(completed.output)) return completed;

  // Fallback: rebuild a minimal completed-response shape from deltas.
  const output: any[] = [];
  if (textOut) output.push({ type: "message", content: [{ type: "output_text", text: textOut }] });
  if (reasoningOut) output.push({ type: "reasoning", summary: reasoningOut });
  for (const slot of Object.values(calls)) {
    output.push({ type: "function_call", call_id: slot.id, name: slot.name, arguments: slot.args || "" });
  }
  return {
    id: createdId,
    object: "response",
    model,
    status: "completed",
    output,
    ...(usage ? { usage } : {}),
  };
}
