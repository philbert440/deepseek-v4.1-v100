#!/usr/bin/env python3
# Smoke test for a DeepSeek-V4.1 server: chat with reasoning, a tool call, and streamed decode tok/s.
import json, sys, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
MODEL = "deepseek-v4.1-flash"


def post(body, stream=False):
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def chat(messages, **kw):
    with post({"model": MODEL, "messages": messages, **kw}) as r:
        return json.load(r)


print("== chat")
t = time.time()
r = chat([{"role": "user", "content": "What is 17 * 23? Answer with just the number."}], max_tokens=400, temperature=0)
m = r["choices"][0]["message"]
print(f"content={m.get('content')!r} reasoning_chars={len(m.get('reasoning') or m.get('reasoning_content') or '')} "
      f"usage={r['usage']} {time.time() - t:.1f}s")

print("== tool call")
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
t = time.time()
r = chat([{"role": "user", "content": "What's the weather in Toronto right now?"}], tools=tools, max_tokens=600, temperature=0)
m = r["choices"][0]["message"]
print(f"tool_calls={json.dumps(m.get('tool_calls'))[:300]} finish={r['choices'][0]['finish_reason']} {time.time() - t:.1f}s")

print("== decode speed (streamed, 256 tokens)")
body = {"model": MODEL, "stream": True, "max_tokens": 256, "temperature": 0, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Write a detailed paragraph about the history of the V100 GPU."}]}
t0 = time.time(); first = None; n = 0; usage = None
with post(body) as r:
    for line in r:
        line = line.decode().strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        ev = json.loads(line[6:])
        if ev.get("usage"):
            usage = ev["usage"]
        for c in ev.get("choices", []):
            d = c.get("delta", {})
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                n += 1
                first = first or time.time()
end = time.time()
toks = (usage or {}).get("completion_tokens", n)
print(f"TTFT={first - t0:.2f}s completion_tokens={toks} decode={(toks - 1) / (end - first):.2f} tok/s total={end - t0:.1f}s")
