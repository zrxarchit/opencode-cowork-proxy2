import { describe, it, expect, vi, afterEach } from 'vitest';
import worker from '../src/index';
import { resetVersionCache, DEFAULT_OPENCODE_VERSION } from '../src/version';

const key = 'a'.repeat(32);

const chatOk = () =>
  new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  });

describe('worker routing', () => {
  afterEach(() => {
    vi.restoreAllMocks();
    resetVersionCache();
  });

  it('lists aliased free models on /v1/models without an API key', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const request = new Request('https://proxy.example/v1/models');

    const response = await worker.fetch(request);
    const body: any = await response.json();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(response.status).toBe(200);
    expect(body.object).toBe('list');
    const ids = body.data.map((m: any) => m.id);
    // Aliases only — real upstream IDs must never leak here.
    expect(ids).toContain('claude-opus-x-1');
    expect(ids).toContain('claude-opus-x-2');
    for (const id of ids) {
      expect(id).not.toContain('-free');
    }
  });

  it('lists aliases in Anthropic format when x-upstream-format is anthropic (no key)', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const request = new Request('https://proxy.example/v1/models', {
      headers: { 'x-upstream-format': 'anthropic' },
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(body.has_more).toBe(false);
    expect(body.data[0].id).toBe('claude-opus-x-1');
  });

  it('serves the public alias map on /v1/map without an API key', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const response = await worker.fetch(new Request('https://proxy.example/v1/map'));
    const body: any = await response.json();

    expect(fetchMock).not.toHaveBeenCalled();
    expect(body['claude-opus-x-1']).toContain('-free');
  });

  it('forwards Anthropic beta header when translating OpenAI requests to Anthropic', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      new Response(JSON.stringify({ content: [{ type: 'text', text: 'ok' }], stop_reason: 'end_turn' }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );

    const request = new Request('https://proxy.example/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'authorization': `Bearer ${key}`,
        'x-upstream-url': 'https://api.anthropic.com',
        'x-upstream-format': 'anthropic',
        'anthropic-beta': 'tools-2024-04-04',
      },
      body: JSON.stringify({ model: 'claude-test', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://api.anthropic.com/v1/messages', expect.objectContaining({
      headers: expect.objectContaining({
        'X-Api-Key': key,
        'Anthropic-Version': '2023-06-01',
        'Anthropic-Beta': 'tools-2024-04-04',
      }),
    }));
  });

  it('routes /go-prefixed Anthropic requests to OpenCode Go', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/go/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'deepseek-v4-pro', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/go/v1/chat/completions', expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Authorization: `Bearer ${key}` }),
    }));
  });

  it('routes /zen-prefixed Anthropic requests to OpenCode Zen', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'qwen3.5-plus', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      method: 'POST',
      headers: expect.objectContaining({ Authorization: `Bearer ${key}` }),
    }));
  });

  it('sends the official OpenCode User-Agent on upstream chat requests', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      headers: expect.objectContaining({ 'User-Agent': expect.stringMatching(/^opencode\//) }),
    }));
  });

  it('passes through the caller User-Agent when it already looks like OpenCode', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': key,
        'user-agent': 'opencode/dev/9.9.9/opencode',
      },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      headers: expect.objectContaining({ 'User-Agent': 'opencode/dev/9.9.9/opencode' }),
    }));
  });

  it('sends OpenCode identity headers on upstream chat requests', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      headers: expect.objectContaining({
        'x-opencode-client': 'cli',
        'x-opencode-session-id': expect.stringMatching(/^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/),
        'x-opencode-session': expect.any(String),
        'x-session-affinity': expect.any(String),
        'x-opencode-request': expect.stringMatching(/^req_/),
      }),
    }));
  });

  it('passes through caller-provided OpenCode identity headers', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': key,
        'x-opencode-session-id': 'ses_customsession123',
        'x-opencode-client': 'opencode',
      },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      headers: expect.objectContaining({ 'x-opencode-session-id': 'ses_customsession123' }),
    }));
  });

  it('forces stream and shell/read tools upstream for free models', async () => {
    let capturedBody: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      capturedBody = JSON.parse(init.body);
      const sse = [
        'data: {"id":"c1","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}',
        'data: {"id":"c1","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
        'data: [DONE]',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();

    // Gate contract enforced upstream even though the client asked for neither.
    expect(capturedBody.stream).toBe(true);
    const toolNames = (capturedBody.tools || []).map((t: any) => t.function?.name);
    expect(toolNames).toContain('shell');
    expect(toolNames).toContain('read');
    // Non-streaming client still gets a single translated message with its alias.
    expect(body.content[0].text).toBe('ok');
    expect(body.model).toBe('claude-opus-x-2');
  });

  it('maps injected shell/read calls back to the client tool names on streams', async () => {
    let capturedBody: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      capturedBody = JSON.parse(init.body);
      const sse = [
        'data: {"id":"t1","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":"{\\"command\\":\\"ls\\"}"}}]},"finish_reason":null}]}',
        'data: {"id":"t1","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        'data: [DONE]',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const bashSchema = { type: 'object', properties: { command: { type: 'string' } }, required: ['command'] };
    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-opus-x-2',
        stream: true,
        tools: [
          { name: 'Bash', description: 'run', input_schema: bashSchema },
          { name: 'Read', description: 'read', input_schema: { type: 'object' } },
        ],
        messages: [{ role: 'user', content: 'list files' }],
      }),
    });

    const response = await worker.fetch(request);
    const text = await response.text();

    // Injected shell mirrors the client's Bash schema upstream...
    const shellTool = capturedBody.tools.find((t: any) => t.function?.name === 'shell');
    expect(shellTool.function.parameters).toEqual(bashSchema);
    // ...and the streamed call comes back renamed to the client's tool.
    expect(text).toContain('"name":"Bash"');
    expect(text).not.toContain('"name":"shell"');
  });

  it('renames injected tool calls on reassembled non-streaming responses', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const sse = [
        'data: {"id":"t2","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_9","type":"function","function":{"name":"read","arguments":"{}"}}]},"finish_reason":null}]}',
        'data: {"id":"t2","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        'data: [DONE]',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-opus-x-2',
        tools: [{ name: 'Read', description: 'read', input_schema: { type: 'object' } }],
        messages: [{ role: 'user', content: 'hi' }],
      }),
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();
    const toolUse = body.content.find((b: any) => b.type === 'tool_use');
    expect(toolUse.name).toBe('Read');
    expect(body.stop_reason).toBe('tool_use');
  });

  it('serves responses-protocol models to Anthropic clients via /responses', async () => {
    let capturedBody: any = null;
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      if (!String(url).includes('api.github.com')) capturedBody = JSON.parse(init.body);
      const sse = [
        'event: response.created',
        'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","model":"muse-spark-1.3-contributor-free","status":"in_progress"}}',
        '',
        'event: response.output_text.delta',
        'data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"delta":"Hello"}',
        '',
        'event: response.completed',
        `data: ${JSON.stringify({ type: "response.completed", sequence_number: 2, response: { id: "resp_1", object: "response", status: "completed", model: "muse-spark-1.3-contributor-free", output: [{ type: "message", id: "msg_1", status: "completed", role: "assistant", content: [{ type: "output_text", text: "Hello", annotations: [] }] }], usage: { input_tokens: 5, output_tokens: 2, total_tokens: 7 } } })}`,
        '',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-1', max_tokens: 30, messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();

    expect(fetchMock).toHaveBeenCalledWith(
      'https://opencode.ai/zen/v1/responses',
      expect.objectContaining({
        method: 'POST',
        headers: expect.objectContaining({ 'x-opencode-client': 'cli' }),
      }),
    );
    expect(body.content[0]).toEqual({ type: 'text', text: 'Hello' });
    expect(body.model).toBe('claude-opus-x-1');
    expect(body.usage.input_tokens).toBe(5);
    // Responses API declares function tools flat (no `function` envelope).
    const toolNames = capturedBody.tools.map((t: any) => t.name);
    expect(toolNames).toContain('shell');
    expect(toolNames).toContain('read');
    for (const t of capturedBody.tools) expect(t.function).toBeUndefined();
    expect(capturedBody.stream).toBe(true);
  });

  it('streams responses-protocol models to streaming Anthropic clients', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const sse = [
        'event: response.output_text.delta',
        'data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_1","output_index":0,"delta":"Hi"}',
        '',
        'event: response.completed',
        'data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_1","object":"response","status":"completed","model":"muse-spark-1.3-contributor-free","output":[],"usage":{"input_tokens":4,"output_tokens":1,"total_tokens":5}}}',
        '',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-1', stream: true, messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const text = await response.text();
    expect(text).toContain('text_delta');
    expect(text).toContain('message_stop');
  });

  it('renames responses function calls back to client tools', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const completed = {
        type: "response.completed",
        sequence_number: 3,
        response: {
          id: "resp_9", object: "response", status: "completed", model: "muse-spark-1.3-contributor-free",
          output: [{ type: "function_call", id: "fc_1", call_id: "call_7", name: "shell", arguments: '{"command":"ls"}' }],
          usage: { input_tokens: 6, output_tokens: 3, total_tokens: 9 },
        },
      };
      return new Response(`event: response.completed\ndata: ${JSON.stringify(completed)}\n\n`, {
        status: 200, headers: { 'Content-Type': 'text/event-stream' },
      });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-opus-x-1',
        tools: [{ name: 'Bash', description: 'run', input_schema: { type: 'object' } }],
        messages: [{ role: 'user', content: 'list files' }],
      }),
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();
    const toolUse = body.content.find((b: any) => b.type === 'tool_use');
    expect(toolUse.name).toBe('Bash');
    expect(toolUse.input).toEqual({ command: 'ls' });
    expect(body.stop_reason).toBe('tool_use');
  });

  it('serves responses-protocol models to OpenAI clients as completions', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const completed = {
        type: "response.completed",
        sequence_number: 2,
        response: {
          id: "resp_2", object: "response", status: "completed", model: "muse-spark-1.3-contributor-free",
          output: [{ type: "message", id: "m1", status: "completed", role: "assistant", content: [{ type: "output_text", text: "ok", annotations: [] }] }],
          usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
        },
      };
      return new Response(`event: response.completed\ndata: ${JSON.stringify(completed)}\n\n`, {
        status: 200, headers: { 'Content-Type': 'text/event-stream' },
      });
    });

    const request = new Request('https://proxy.example/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'authorization': `Bearer ${key}` },
      body: JSON.stringify({ model: 'claude-opus-x-1', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();
    expect(body.object).toBe('chat.completion');
    expect(body.choices[0].message.content).toBe('ok');
    expect(body.model).toBe('claude-opus-x-1');
  });

  it('serves responses-protocol models to streaming OpenAI clients as one chunk', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const completed = {
        type: "response.completed",
        sequence_number: 2,
        response: {
          id: "resp_3", object: "response", status: "completed", model: "muse-spark-1.3-contributor-free",
          output: [{ type: "message", id: "m1", status: "completed", role: "assistant", content: [{ type: "output_text", text: "ok", annotations: [] }] }],
          usage: { input_tokens: 3, output_tokens: 1, total_tokens: 4 },
        },
      };
      return new Response(`event: response.completed\ndata: ${JSON.stringify(completed)}\n\n`, {
        status: 200, headers: { 'Content-Type': 'text/event-stream' },
      });
    });

    const request = new Request('https://proxy.example/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'authorization': `Bearer ${key}` },
      body: JSON.stringify({ model: 'claude-opus-x-1', stream: true, messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const text = await response.text();
    expect(response.headers.get('Content-Type')).toContain('text/event-stream');
    expect(text).toContain('chat.completion.chunk');
    expect(text).toContain('[DONE]');
  });

  it('uses only the targeted model: surfaces its error without retrying others', async () => {
    const seenModels: string[] = [];
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      seenModels.push(JSON.parse(init.body).model);
      return new Response('{"type":"error","error":{"type":"FreeUsageLimitError","message":"Rate limit exceeded"}}', {
        status: 429, headers: { 'Content-Type': 'application/json' },
      });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    expect(response.status).toBe(429);
    // Exactly one upstream attempt, against the targeted model only.
    const upstreamCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('opencode.ai'));
    expect(upstreamCalls).toHaveLength(1);
    expect(seenModels).toHaveLength(1);
    expect(seenModels[0]).toContain('-free');
  });

  it('makes a single upstream attempt for paid and pinned URL models alike', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      return new Response('{"error":"limited"}', {
        status: 429, headers: { 'Content-Type': 'application/json' },
      });
    });

    for (const url of [
      'https://proxy.example/zen/v1/messages',
      'https://proxy.example/zen/mimo-v2.6-flash-free/v1/messages',
    ]) {
      const request = new Request(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': key },
        body: JSON.stringify({
          model: url.includes('/mimo-') ? 'claude-sonnet-4-5-20250514' : 'deepseek-v4-pro',
          messages: [{ role: 'user', content: 'hi' }],
        }),
      });
      const response = await worker.fetch(request);
      expect(response.status).toBe(429);
    }
    const upstreamCalls = fetchMock.mock.calls.filter(([u]) => String(u).includes('opencode.ai'));
    expect(upstreamCalls).toHaveLength(2);
  });

  it('makes a single upstream attempt on the OpenAI passthrough path', async () => {
    const seenModels: string[] = [];
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      seenModels.push(JSON.parse(init.body).model);
      return new Response('{"error":"busy"}', {
        status: 503, headers: { 'Content-Type': 'application/json' },
      });
    });

    const request = new Request('https://proxy.example/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'authorization': `Bearer ${key}` },
      body: JSON.stringify({ model: 'claude-opus-x-2', stream: true, messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    expect(response.status).toBe(503);
    const upstreamCalls = fetchMock.mock.calls.filter(([u]) => String(u).includes('opencode.ai'));
    expect(upstreamCalls).toHaveLength(1);
    expect(seenModels).toHaveLength(1);
  });

  it('rejects responses-protocol models on Anthropic-native upstreams with a clear error', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    const request = new Request('https://proxy.example/v1/chat/completions', {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'authorization': `Bearer ${key}`,
        'x-upstream-url': 'https://api.anthropic.com',
        'x-upstream-format': 'anthropic',
      },
      body: JSON.stringify({ model: 'claude-opus-x-1', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    expect(response.status).toBe(400);
    const upstreamCalls = fetchMock.mock.calls.filter(([url]) => !String(url).includes('api.github.com'));
    expect(upstreamCalls).toHaveLength(0);
  });

  it('streams provider reasoning deltas as thinking blocks', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const sse = [
        'data: {"id":"t3","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"let me think"},"finish_reason":null}]}',
        'data: {"id":"t3","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}',
        'data: [DONE]',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', stream: true, messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const text = await response.text();
    expect(text).toContain('thinking_delta');
    expect(text).toContain('let me think');
  });

  it('leaves paid models untouched (no forced stream or tools)', async () => {
    let capturedBody: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      capturedBody = JSON.parse(init.body);
      return chatOk();
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'deepseek-v4-pro', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request);

    expect(capturedBody.stream).toBeUndefined();
    expect(capturedBody.tools).toBeUndefined();
  });

  it('reassembles one completion for non-streaming OpenAI clients on free models', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } });
      }
      const sse = [
        'data: {"id":"c2","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},"finish_reason":null}]}',
        'data: {"id":"c2","model":"mimo-v2.6-flash-free","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}}',
        'data: [DONE]',
      ].join('\n');
      return new Response(sse, { status: 200, headers: { 'Content-Type': 'text/event-stream' } });
    });

    const request = new Request('https://proxy.example/v1/chat/completions', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'authorization': `Bearer ${key}` },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);
    const body: any = await response.json();

    expect(body.object).toBe('chat.completion');
    expect(body.choices[0].message.content).toBe('hi');
    expect(body.model).toBe('claude-opus-x-2');
  });

  it('builds the upstream User-Agent from the latest stable GitHub release', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any, init: any) => {
      if (String(url).includes('api.github.com')) {
        expect(init.headers.Authorization).toBe('Bearer test-github-key');
        return new Response(JSON.stringify({ tag_name: 'v9.9.9' }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      }
      return chatOk();
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    await worker.fetch(request, { GITHUB_API_KEY: 'test-github-key' });

    expect(fetchMock).toHaveBeenCalledWith(
      'https://api.github.com/repos/anomalyco/opencode/releases/latest',
      expect.anything(),
    );
    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      headers: expect.objectContaining({ 'User-Agent': 'opencode/stable/9.9.9/opencode' }),
    }));
  });

  it('falls back to the built-in version when GitHub is unreachable', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(async (url: any) => {
      if (String(url).includes('api.github.com')) {
        return new Response('rate limited', { status: 403 });
      }
      return chatOk();
    });

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'claude-opus-x-2', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);

    expect(response.status).toBe(200);
    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.objectContaining({
      headers: expect.objectContaining({ 'User-Agent': `opencode/stable/${DEFAULT_OPENCODE_VERSION}/opencode` }),
    }));
  });

  it('preserves upstream rate limit headers on translated errors', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () =>
      new Response('{"error":"FreeUsageLimitError"}', {
        status: 429,
        headers: {
          'Content-Type': 'application/json',
          'Retry-After': '60',
          'RateLimit-Reset': '1710000000',
        },
      }),
    );

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({ model: 'minimax-m2.5-free', messages: [{ role: 'user', content: 'hi' }] }),
    });

    const response = await worker.fetch(request);

    expect(response.status).toBe(429);
    expect(response.headers.get('Content-Type')).toContain('application/json');
    expect(response.headers.get('Retry-After')).toBe('60');
    expect(response.headers.get('RateLimit-Reset')).toBe('1710000000');
    expect(await response.text()).toBe('{"error":"FreeUsageLimitError"}');
  });

  it('serves the alias list on /go/v1/models and /zen/v1/models without an API key', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    for (const url of ['https://proxy.example/go/v1/models', 'https://proxy.example/zen/v1/models']) {
      const response = await worker.fetch(new Request(url));
      const body: any = await response.json();
      expect(response.status).toBe(200);
      expect(body.data.length).toBeGreaterThan(0);
      expect(body.data[0].id).toBe('claude-opus-x-1');
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('serves the alias map on /go/v1/map and /zen/v1/map without an API key', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch');

    for (const url of ['https://proxy.example/go/v1/map', 'https://proxy.example/zen/v1/map']) {
      const response = await worker.fetch(new Request(url));
      const body: any = await response.json();
      expect(body['claude-opus-x-1']).toContain('-free');
    }
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('resolves an alias to the real free model on Anthropic requests', async () => {
    let capturedBody: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async (_url, init: any) => {
        capturedBody = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
    );

    const request = new Request('https://proxy.example/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-opus-x-1',
        messages: [{ role: 'user', content: 'hi' }],
      }),
    });

    const response = await worker.fetch(request);
    const body = await response.json();
    // Upstream gets the real free model...
    expect(capturedBody.model).toContain('-free');
    // ...but the client sees its alias echoed back.
    expect(body.model).toBe('claude-opus-x-1');
  });

  it('overrides model from URL path segment with /go prefix', async () => {
    let capturedBody: any = null;
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(
      async (_url, init: any) => {
        capturedBody = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
    );

    const request = new Request('https://proxy.example/go/minimax-m2.5-free/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-sonnet-4-5-20250514',
        messages: [{ role: 'user', content: 'hi' }],
      }),
    });

    const response = await worker.fetch(request);
    expect(capturedBody.model).toBe('minimax-m2.5-free');
    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/go/v1/chat/completions', expect.anything());
  });

  it('overrides model from URL path segment with /zen prefix', async () => {
    let capturedBody: any = null;
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(
      async (_url, init: any) => {
        capturedBody = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
    );

    const request = new Request('https://proxy.example/zen/minimax-m2.5-free/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-sonnet-4-5-20250514',
        messages: [{ role: 'user', content: 'hi' }],
      }),
    });

    await worker.fetch(request);
    expect(capturedBody.model).toBe('minimax-m2.5-free');
    expect(fetchMock).toHaveBeenCalledWith('https://opencode.ai/zen/v1/chat/completions', expect.anything());
  });

  it('returns original model name in response body when model override is active', async () => {
    vi.spyOn(globalThis, 'fetch').mockImplementation(async () => chatOk());

    const request = new Request('https://proxy.example/go/minimax-m2.5-free/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'claude-sonnet-4-5-20250514',
        messages: [{ role: 'user', content: 'hi' }],
      }),
    });

    const response = await worker.fetch(request);
    const body = await response.json();
    expect(body.model).toBe('claude-sonnet-4-5-20250514');
  });

  it('does not override model when no model segment in path', async () => {
    let capturedBody: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async (_url, init: any) => {
        capturedBody = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
    );

    const request = new Request('https://proxy.example/go/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'deepseek-v4-pro',
        messages: [{ role: 'user', content: 'hi' }],
      }),
    });

    await worker.fetch(request);
    expect(capturedBody.model).toBe('deepseek-v4-pro');
  });

  it('overrides model to qwen3.6-plus when image attachments are present on the go path', async () => {
    let capturedBody: any = null;
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockImplementation(
      async (_url: any, init: any) => {
        if (init?.body) capturedBody = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
    );

    const request = new Request('https://proxy.example/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'deepseek-v4-pro',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is in this image?' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc123' } },
          ],
        }],
        max_tokens: 1024,
      }),
    });

    await worker.fetch(request);
    const upstreamCalls = fetchMock.mock.calls.filter(([url]) => String(url).includes('opencode.ai'));
    expect(upstreamCalls).toHaveLength(1);
    expect(capturedBody.model).toBe('qwen3.6-plus');
    expect(Array.isArray(capturedBody.messages[0].content)).toBe(true);
    expect(capturedBody.messages[0].content).toEqual([
      { type: 'text', text: 'What is in this image?' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,abc123' } },
    ]);
  });

  it('overrides model to qwen3.6-plus when image attachments are present on the zen path', async () => {
    let capturedBody: any = null;
    vi.spyOn(globalThis, 'fetch').mockImplementation(
      async (_url, init: any) => {
        capturedBody = JSON.parse(init.body);
        return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }), {
          status: 200,
          headers: { 'Content-Type': 'application/json' },
        });
      },
    );

    const request = new Request('https://proxy.example/zen/v1/messages', {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': key },
      body: JSON.stringify({
        model: 'mimo-v2.5-free',
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is in this image?' },
            { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'abc123' } },
          ],
        }],
        max_tokens: 1024,
      }),
    });

    await worker.fetch(request);
    expect(capturedBody.model).toBe('qwen3.6-plus');
  });
});
