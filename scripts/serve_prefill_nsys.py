#!/usr/bin/env python3
"""One long-prompt prefill request, for nsys profiling (no decode tail).

Usage: serve_prefill_nsys.py HOST PORT [PROMPT_FILE]
PROMPT_FILE defaults to /tmp/prompt_2000.txt on the host.
"""
import http.client
import json
import sys
import time

from serve_client import served_model

HOST = sys.argv[1]
PORT = int(sys.argv[2])
PROMPT_FILE = sys.argv[3] if len(sys.argv) > 3 else "/tmp/prompt_2000.txt"

prompt = open(PROMPT_FILE).read()
body = json.dumps({
    "model": served_model(HOST, PORT),
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": 1,
    "temperature": 0,
})
t0 = time.monotonic()
conn = http.client.HTTPConnection(HOST, PORT, timeout=300)
conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
resp = conn.getresponse()
out = json.loads(resp.read())
dt = time.monotonic() - t0
usage = out.get("usage", {})
print(f"prompt_chars={len(prompt)} prompt_tokens={usage.get('prompt_tokens')} "
      f"completion_tokens={usage.get('completion_tokens')} client_e2e={dt:.3f} s")
