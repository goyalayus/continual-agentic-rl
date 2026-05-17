#!/usr/bin/env python3
"""Preflight provider calls needed by the full banking benchmark."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


CHAT_MODEL = "azure/gpt-5.5"
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
REPO_ROOT = Path(__file__).resolve().parents[2]
CUSTOM_HARNESS_DIR = REPO_ROOT / "custom_harness"
if str(CUSTOM_HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_HARNESS_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-model", default=CHAT_MODEL)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args()


def require_azure_chat_env() -> None:
    api_key = os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
    api_base = os.environ.get("AZURE_API_BASE") or os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_version = os.environ.get("AZURE_API_VERSION") or os.environ.get(
        "AZURE_OPENAI_API_VERSION"
    )
    missing = []
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY/AZURE_API_KEY")
    if not api_base:
        missing.append("AZURE_OPENAI_ENDPOINT/AZURE_API_BASE")
    if not api_version:
        missing.append("AZURE_OPENAI_API_VERSION/AZURE_API_VERSION")
    if missing:
        raise RuntimeError("missing Azure chat env: " + ", ".join(missing))

    os.environ.setdefault("AZURE_API_KEY", api_key)
    os.environ.setdefault("AZURE_API_BASE", api_base)
    os.environ.setdefault("AZURE_API_VERSION", api_version)


def require_openrouter_embedding_key() -> str:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set for Qwen query embeddings")
    if not api_key.startswith("sk-or-"):
        raise RuntimeError(
            "OPENROUTER_API_KEY does not look like an OpenRouter key; "
            "expected a value starting with sk-or-"
        )
    return api_key


def preflight_chat(args: argparse.Namespace) -> None:
    from litellm import responses

    response = responses(
        model=args.chat_model,
        input=[{"role": "user", "content": "Reply OK only."}],
        max_output_tokens=args.max_tokens,
        reasoning={"effort": args.reasoning_effort},
        api_base=os.environ.get("AZURE_API_BASE"),
        api_version=os.environ.get("AZURE_API_VERSION"),
    )
    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    content_parts = []
    for item in raw.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            if part.get("text"):
                content_parts.append(part["text"])
    content = "\n".join(content_parts)
    print(f"chat_ok model={args.chat_model} response={content[:40]!r}")


def preflight_hybrid_retrieval(args: argparse.Namespace) -> None:
    from tau3_custom_harness.retrieval import BankingHybridRetriever

    retriever = BankingHybridRetriever(embedding_model=args.embedding_model)
    query = "hybrid retrieval preflight"
    embedding = retriever._query_embedding(query)
    if embedding is None:
        raise RuntimeError(
            "hybrid retriever preflight could not fetch a query embedding; "
            "the custom harness would silently degrade to BM25-only"
        )
    if embedding.shape[0] != retriever.embeddings.shape[1]:
        raise RuntimeError(
            "hybrid retriever preflight embedding dimension mismatch "
            f"received={embedding.shape[0]} expected={retriever.embeddings.shape[1]}"
        )
    hits = retriever.search(query, top_k=1)
    if not hits:
        raise RuntimeError(
            "hybrid retriever preflight returned no hits; the custom harness "
            "would likely degrade to BM25-only or fail retrieval"
        )
    print(
        "hybrid_retrieval_ok "
        f"model={args.embedding_model} "
        f"dims={retriever.embeddings.shape[1]} "
        f"top_doc={hits[0].doc_id!r}"
    )


def main() -> int:
    args = parse_args()
    try:
        require_azure_chat_env()
        if not args.skip_embeddings:
            require_openrouter_embedding_key()
        preflight_chat(args)
        if not args.skip_embeddings:
            preflight_hybrid_retrieval(args)
    except Exception as exc:
        print(
            f"preflight_failed {type(exc).__name__}: {sanitize_error(str(exc))}",
            file=sys.stderr,
        )
        return 1
    print("provider_preflight_ok")
    return 0


def sanitize_error(text: str) -> str:
    text = re.sub(r'("user_id"\s*:\s*")[^"]+(")', r"\1[REDACTED]\2", text)
    text = re.sub(r"sk-or-v1-[A-Za-z0-9_-]+", "[REDACTED_OPENROUTER_KEY]", text)
    text = re.sub(r"sk-proj-[A-Za-z0-9_-]+", "[REDACTED_OPENAI_KEY]", text)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
