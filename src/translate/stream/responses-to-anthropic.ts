/**
 * Translates an OpenAI Responses API SSE stream to Anthropic Messages SSE.
 */

export function streamResponsesToAnthropic(
  responsesStream: ReadableStream,
  model: string,
  toolNameMap: Record<string, string> = {},
): ReadableStream {
  const messageId = `msg_${Date.now()}`;

  const enqueueSSE = (controller: ReadableStreamDefaultController, eventType: string, data: any) => {
    controller.enqueue(new TextEncoder().encode(`event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`));
  };

  return new ReadableStream({
    async start(controller) {
      let contentBlockIndex = -1;
      let messageStarted = false;
      let sawToolCall = false;
      let lastUsage: any = null;
      const textBlockByOutput = new Map<number, number>();
      const toolBlockByItem = new Map<string, number>();
      let thinkingBlock: number | null = null;

      const ensureMessageStart = () => {
        if (messageStarted) return;
        enqueueSSE(controller, "message_start", {
          type: "message_start",
          message: {
            id: messageId,
            type: "message",
            role: "assistant",
            content: [],
            model,
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 },
          },
        });
        messageStarted = true;
      };

      const openTextBlock = (outputIndex: number): number => {
        const existing = textBlockByOutput.get(outputIndex);
        if (existing !== undefined) return existing;
        contentBlockIndex++;
        textBlockByOutput.set(outputIndex, contentBlockIndex);
        ensureMessageStart();
        enqueueSSE(controller, "content_block_start", {
          type: "content_block_start",
          index: contentBlockIndex,
          content_block: { type: "text", text: "" },
        });
        return contentBlockIndex;
      };

      const openThinkingBlock = (): number => {
        if (thinkingBlock !== null) return thinkingBlock;
        contentBlockIndex++;
        thinkingBlock = contentBlockIndex;
        ensureMessageStart();
        enqueueSSE(controller, "content_block_start", {
          type: "content_block_start",
          index: thinkingBlock,
          content_block: { type: "thinking", thinking: "", signature: "" },
        });
        return thinkingBlock;
      };

      const processEvent = (event: any) => {
        const type = event?.type;
        if (!type) return;

        if (type === "response.output_text.delta" && typeof event.delta === "string") {
          const index = openTextBlock(event.output_index ?? 0);
          enqueueSSE(controller, "content_block_delta", {
            type: "content_block_delta",
            index,
            delta: { type: "text_delta", text: event.delta },
          });
        } else if (type === "response.reasoning_summary_text.delta" && typeof event.delta === "string") {
          const index = openThinkingBlock();
          enqueueSSE(controller, "content_block_delta", {
            type: "content_block_delta",
            index,
            delta: { type: "thinking_delta", thinking: event.delta },
          });
        } else if (type === "response.output_item.added") {
          const item = event.item || {};
          if (item.type === "function_call") {
            const key = item.call_id || item.id || `item-${event.output_index ?? 0}`;
            if (!toolBlockByItem.has(key)) {
              contentBlockIndex++;
              toolBlockByItem.set(key, contentBlockIndex);
              ensureMessageStart();
              enqueueSSE(controller, "content_block_start", {
                type: "content_block_start",
                index: contentBlockIndex,
                content_block: {
                  type: "tool_use",
                  id: item.call_id || item.id,
                  name: toolNameMap[item.name] ?? item.name,
                  input: {},
                },
              });
              sawToolCall = true;
            }
          }
        } else if (type === "response.function_call_arguments.delta" && typeof event.delta === "string") {
          const key = event.item_id;
          const index = key !== undefined ? toolBlockByItem.get(key) : undefined;
          if (index !== undefined) {
            enqueueSSE(controller, "content_block_delta", {
              type: "content_block_delta",
              index,
              delta: { type: "input_json_delta", partial_json: event.delta },
            });
          }
        } else if ((type === "response.completed" || type === "response.incomplete") && event.response?.usage) {
          lastUsage = {
            input_tokens: event.response.usage.input_tokens ?? 0,
            output_tokens: event.response.usage.output_tokens ?? 0,
          };
        }
      };

      const reader = responsesStream.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      const handleLines = (lines: string[]) => {
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data:")) continue;
          const data = trimmed.slice(5).trim();
          if (!data || data === "[DONE]") continue;
          try {
            processEvent(JSON.parse(data));
          } catch {
            // parse error
          }
        }
      };

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (buffer.trim()) handleLines(buffer.split("\n"));
            break;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";
          handleLines(lines);
        }
      } finally {
        reader.releaseLock();
      }

      if (contentBlockIndex >= 0) {
        enqueueSSE(controller, "content_block_stop", {
          type: "content_block_stop",
          index: contentBlockIndex,
        });
      }
      enqueueSSE(controller, "message_delta", {
        type: "message_delta",
        delta: { stop_reason: sawToolCall ? "tool_use" : "end_turn", stop_sequence: null },
        usage: lastUsage || { input_tokens: 0, output_tokens: 0 },
      });
      enqueueSSE(controller, "message_stop", { type: "message_stop" });
      controller.close();
    },
  });
}
