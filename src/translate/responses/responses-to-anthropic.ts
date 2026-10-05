/**
 * Converts a completed OpenAI Responses API object to Anthropic Messages format.
 */

function parseToolArguments(value: string | undefined): Record<string, unknown> {
  if (!value) return {};
  try {
    const parsed = JSON.parse(value);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function reasoningTextOf(item: any): string {
  const summary = item.summary;
  if (typeof summary === "string") return summary;
  if (Array.isArray(summary)) {
    return summary
      .map((s: any) => (typeof s === "string" ? s : s?.text))
      .filter((s: any) => typeof s === "string")
      .join("\n");
  }
  if (typeof item.content === "string") return item.content;
  return "";
}

export function formatResponsesToAnthropic(response: any, model: string, toolNameMap: Record<string, string> = {}): any {
  const content: any[] = [];
  let sawToolCall = false;

  for (const item of response?.output || []) {
    if (item.type === "message") {
      for (const part of item.content || []) {
        if (part.type === "output_text" && typeof part.text === "string") {
          content.push({ type: "text", text: part.text });
        } else if (part.type === "refusal" && typeof part.refusal === "string") {
          content.push({ type: "text", text: part.refusal });
        }
      }
    } else if (item.type === "function_call") {
      sawToolCall = true;
      content.push({
        type: "tool_use",
        id: item.call_id || item.id,
        name: toolNameMap[item.name] ?? item.name,
        input: parseToolArguments(item.arguments),
      });
    } else if (item.type === "reasoning") {
      const text = reasoningTextOf(item);
      if (text) content.push({ type: "thinking", thinking: text, signature: "" });
    }
  }

  const result: any = {
    id: `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    content,
    stop_reason: sawToolCall ? "tool_use" : "end_turn",
    stop_sequence: null,
    model,
  };

  const usage = response?.usage;
  if (usage) {
    result.usage = {
      input_tokens: usage.input_tokens ?? 0,
      output_tokens: usage.output_tokens ?? 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    };
  }
  return result;
}

/**
 * Converts a completed Responses object to OpenAI chat-completion format
 * (for OpenAI clients talking to responses-protocol models through this proxy).
 */
export function formatResponsesToChatCompletion(response: any, model: string, toolNameMap: Record<string, string> = {}): any {
  let content: string | null = null;
  let reasoning = "";
  const toolCalls: any[] = [];

  for (const item of response?.output || []) {
    if (item.type === "message") {
      for (const part of item.content || []) {
        if ((part.type === "output_text" || part.type === "refusal") && typeof (part.text ?? part.refusal) === "string") {
          const text = part.text ?? part.refusal;
          content = (content || "") + text;
        }
      }
    } else if (item.type === "function_call") {
      toolCalls.push({
        id: item.call_id || item.id,
        type: "function",
        function: {
          name: toolNameMap[item.name] ?? item.name,
          arguments: typeof item.arguments === "string" ? item.arguments : "{}",
        },
      });
    } else if (item.type === "reasoning") {
      const text = reasoningTextOf(item);
      if (text) reasoning += text;
    }
  }

  const message: any = { role: "assistant", content };
  if (reasoning) message.reasoning_content = reasoning;
  if (toolCalls.length > 0) message.tool_calls = toolCalls;

  return {
    id: response?.id || `chatcmpl-${Date.now()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, message, finish_reason: toolCalls.length > 0 ? "tool_calls" : "stop" }],
    ...(response?.usage
      ? {
          usage: {
            prompt_tokens: response.usage.input_tokens ?? 0,
            completion_tokens: response.usage.output_tokens ?? 0,
            total_tokens: response.usage.total_tokens ?? 0,
          },
        }
      : {}),
  };
}
