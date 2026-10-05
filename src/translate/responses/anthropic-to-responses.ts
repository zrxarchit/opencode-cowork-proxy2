/**
 * Converts an Anthropic Messages request to an OpenAI Responses API request.
 * Used for models that only speak the Responses protocol
 * (e.g. `muse-spark-1.3-contributor-free` on OpenCode Zen).
 */

function textOf(value: any): string {
  return typeof value === "string" ? value : JSON.stringify(value);
}

function imageUrlOf(source: any): string | null {
  if (!source) return null;
  if (source.type === "url") return source.url;
  if (source.type === "base64") return `data:${source.media_type};base64,${source.data}`;
  return null;
}

export function formatAnthropicToResponses(body: any): any {
  const { model, messages, system, temperature, max_tokens, top_p, tools, stream } = body;

  const instructions = Array.isArray(system)
    ? system.map((item: any) => item.text).join("\n\n")
    : system;

  const input: any[] = [];
  for (const msg of messages || []) {
    if (typeof msg.content === "string") {
      input.push({
        role: msg.role,
        content: [{ type: msg.role === "assistant" ? "output_text" : "input_text", text: msg.content }],
      });
      continue;
    }
    if (!Array.isArray(msg.content)) continue;

    if (msg.role === "user") {
      const parts: any[] = [];
      for (const part of msg.content) {
        if (part.type === "text") {
          parts.push({ type: "input_text", text: textOf(part.text) });
        } else if (part.type === "image") {
          const url = imageUrlOf(part.source);
          if (url) parts.push({ type: "input_image", image_url: url });
        } else if (part.type === "tool_result") {
          input.push({
            type: "function_call_output",
            call_id: part.tool_use_id,
            output: textOf(part.content),
          });
        }
      }
      if (parts.length > 0) input.push({ role: "user", content: parts });
    } else if (msg.role === "assistant") {
      const parts: any[] = [];
      for (const part of msg.content) {
        if (part.type === "text") {
          parts.push({ type: "output_text", text: textOf(part.text) });
        } else if (part.type === "tool_use") {
          input.push({
            type: "function_call",
            call_id: part.id,
            name: part.name,
            arguments: JSON.stringify(part.input ?? {}),
          });
        }
        // thinking blocks cannot be sent back; dropped.
      }
      if (parts.length > 0) input.push({ role: "assistant", content: parts });
    }
  }

  const req: any = { model, input };
  if (instructions) req.instructions = instructions;
  if (stream !== undefined) req.stream = stream;
  if (max_tokens !== undefined) req.max_output_tokens = max_tokens;
  if (temperature !== undefined) req.temperature = temperature;
  if (top_p !== undefined) req.top_p = top_p;
  if (tools) {
    req.tools = tools.map((item: any) => ({
      type: "function",
      name: item.name,
      description: item.description,
      parameters: item.input_schema,
    }));
  }
  return req;
}
