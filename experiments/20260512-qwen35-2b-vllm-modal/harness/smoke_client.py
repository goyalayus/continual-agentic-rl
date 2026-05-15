#!/usr/bin/env python3
"""Tiny OpenAI-compatible client for the Modal vLLM server."""

from __future__ import annotations

import argparse
import json
import urllib.request
from typing import Any


def post_json(url: str, payload: dict[str, Any], *, api_key: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True, help="Server root, without /v1.")
    parser.add_argument("--model", default="qwen3.5-2b")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
        "temperature": 0.0,
        "max_tokens": 16,
    }
    response = post_json(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        payload,
        api_key=args.api_key,
        timeout=args.timeout,
    )
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
