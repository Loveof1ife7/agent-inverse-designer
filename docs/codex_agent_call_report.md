# Codex Agent Local Endpoint Call Report

Date: 2026-06-14 UTC
Workspace: `/root/autodl-tmp/projects/agent-material`

## Purpose

This report records the verified way to call the local OpenAI-compatible Codex endpoint from another agent or script.

## Verified Connection Settings

Base URL:

```text
http://127.0.0.1:17899/v1
```

API key:

```text
e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db
```

Recommended model:

```text
gpt-5.5
```

## Important Runtime Note

This machine has proxy settings that can hijack local requests. Direct calls must bypass system proxies, or the request may be redirected to another local port and fail.

For Python, the local demo already disables proxies internally:

```text
/root/autodl-tmp/projects/agent-material/codex_agent_demo.py
```

For shell `curl`, use `--noproxy '*'` and unset proxy environment variables.

## Verified API Behavior

`GET /v1/models` succeeded and returned these visible models:

```text
gpt-5.5
gpt-5.4
gpt-5.4-mini
gpt-5.3-codex
gpt-5.2
codex-auto-review
gpt-image-2
```

`POST /v1/responses` with `model="gpt-5.5"` succeeded multiple times.

Verified outputs:

```text
Prompt: 请用一句中文说明你已通过本地 OpenAI 兼容端点被成功调用。
Output: 我已通过本地 OpenAI 兼容端点被成功调用。
```

```text
Prompt: 把 37 和 58 相加，只输出数字。
Output: 95
```

```text
Prompt: 输出一个单行 JSON：{"status":"ok","model":"gpt-5.5"}
Output: {"status":"ok","model":"gpt-5.5"}
```

## Known Limitation

`gpt-5.3-codex` appears in `/v1/models`, but an actual `/v1/responses` call was rejected by the server with this message:

```text
The 'gpt-5.3-codex' model is not supported when using Codex with a ChatGPT account.
```

Conclusion:

```text
Use gpt-5.5 for reliable calls on this setup.
Do not assume gpt-5.3-codex is usable just because it is listed.
```

## Reproduction Commands

List models:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
curl --noproxy '*' -sS \
  -H 'Authorization: Bearer e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db' \
  'http://127.0.0.1:17899/v1/models'
```

Minimal successful call:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy \
curl --noproxy '*' -sS \
  -H 'Authorization: Bearer e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db' \
  -H 'Content-Type: application/json' \
  'http://127.0.0.1:17899/v1/responses' \
  -d '{"model":"gpt-5.5","input":"只回复 pong"}'
```

Python demo:

```bash
python /root/autodl-tmp/projects/agent-material/codex_agent_demo.py \
  --api-key 'e339901aa965318c253573770ae65bd0cfce8b93ca057ddc79725f80186382db' \
  --base-url 'http://127.0.0.1:17899/v1' \
  --model gpt-5.5 \
  '只回复 pong'
```

## Recommended Instructions For Another Agent

Use the local OpenAI-compatible endpoint at `http://127.0.0.1:17899/v1` with the provided API key.

Default to `gpt-5.5`.

Bypass proxies for all local requests.

If the task only needs a quick connectivity check, call `/v1/models` first, then verify `/v1/responses`.

If `gpt-5.3-codex` is requested, expect a server-side account restriction unless the account configuration changes.

## Security Note

This report contains a live API key. If it is shared beyond the intended local/trusted agent workflow, rotate the key.

## Example
```

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Iterator


DEFAULT_BASE_URL = "http://127.0.0.1:17899/v1"
DEFAULT_MODEL = "gpt-5.5"
NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal Codex/OpenAI-compatible local endpoint demo."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain this demo in one sentence.",
        help="Prompt sent to the /responses API.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", ""),
        help="API key. Defaults to OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("OPENAI_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible base URL. Defaults to {DEFAULT_BASE_URL}.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", DEFAULT_MODEL),
        help=f"Model name. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        default="medium",
        help="Reasoning effort passed to the responses API.",
    )
    return parser.parse_args()


def make_request(
    url: str,
    api_key: str,
    payload: dict | None = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with NO_PROXY_OPENER.open(request, timeout=120) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed: {exc}") from exc


def iter_sse_events(raw_text: str) -> Iterator[tuple[str, str]]:
    event_name = ""
    data_lines: list[str] = []
    for line in raw_text.splitlines():
        if not line:
            if data_lines:
                yield event_name or "message", "\n".join(data_lines)
                event_name = ""
                data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
            continue
        if line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())
    if data_lines:
        yield event_name or "message", "\n".join(data_lines)


def list_models(base_url: str, api_key: str) -> int:
    body = make_request(f"{base_url.rstrip('/')}/models", api_key)
    payload = json.loads(body)
    models = payload.get("data", [])
    if not models:
        print("No models returned.")
        return 0
    for model in models:
        print(model.get("id", "<missing-id>"))
    return 0


def run_prompt(base_url: str, api_key: str, model: str, prompt: str, reasoning_effort: str) -> int:
    payload = {
        "model": model,
        "input": prompt,
        "reasoning": {"effort": reasoning_effort},
    }
    raw_text = make_request(f"{base_url.rstrip('/')}/responses", api_key, payload)

    printed_text = False
    for event_name, data in iter_sse_events(raw_text):
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue

        if event_name == "response.output_text.delta":
            sys.stdout.write(payload.get("delta", ""))
            sys.stdout.flush()
            printed_text = True
            continue

        if event_name == "response.failed":
            error = payload.get("response", {}).get("error") or payload.get("error") or payload
            raise RuntimeError(json.dumps(error, ensure_ascii=False))

    if printed_text:
        sys.stdout.write("\n")
        return 0

    fallback_texts: list[str] = []
    for event_name, data in iter_sse_events(raw_text):
        if not event_name.endswith(".done"):
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        text = payload.get("text")
        if isinstance(text, str) and text:
            fallback_texts.append(text)

    if fallback_texts:
        print("".join(fallback_texts))
        return 0

    print(raw_text)
    return 0


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Missing API key. Pass --api-key or set OPENAI_API_KEY.", file=sys.stderr)
        return 2

    try:
        if args.list_models:
            return list_models(args.base_url, args.api_key)
        return run_prompt(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            prompt=args.prompt,
            reasoning_effort=args.reasoning_effort,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


```