#!/usr/bin/env python3
"""Build a chunk-level embedding index for the banking knowledge base.

The artifact is intentionally standalone so the custom harness can use it
without depending on Tau's current retrieval tool implementations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS_DIR = (
    REPO_ROOT / "data" / "tau2" / "domains" / "banking_knowledge" / "documents"
)
DEFAULT_OUT_DIR = (
    REPO_ROOT
    / "data"
    / ".embeddings_cache"
    / "banking_knowledge"
    / "custom_chunk_index_qwen3_8b"
)
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="qwen/qwen3-embedding-8b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-words", type=int, default=550)
    parser.add_argument("--overlap-words", type=int, default=75)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    return parser.parse_args()


def load_documents(docs_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for path in sorted(docs_dir.glob("*.json")):
        with path.open() as f:
            raw = json.load(f)
        docs.append(
            {
                "doc_id": raw["id"],
                "title": raw["title"],
                "content": raw["content"],
                "source_file": str(path.relative_to(REPO_ROOT)),
            }
        )
    if not docs:
        raise RuntimeError(f"No document JSON files found in {docs_dir}")
    return docs


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def chunk_long_block(block: str, max_words: int, overlap_words: int) -> list[str]:
    words = block.split()
    if len(words) <= max_words:
        return [block]

    chunks = []
    step = max(1, max_words - overlap_words)
    for start in range(0, len(words), step):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return chunks


def split_into_chunks(content: str, max_words: int, overlap_words: int) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_words = 0

    for block in blocks:
        block_words = word_count(block)
        if block_words > max_words:
            flush()
            chunks.extend(chunk_long_block(block, max_words, overlap_words))
            continue

        if current and current_words + block_words > max_words:
            flush()

        current.append(block)
        current_words += block_words

    flush()
    return chunks or [content.strip()]


def build_chunks(
    docs: list[dict[str, Any]], max_words: int, overlap_words: int
) -> list[dict[str, Any]]:
    chunks = []
    for doc in docs:
        doc_chunks = split_into_chunks(doc["content"], max_words, overlap_words)
        for index, text in enumerate(doc_chunks):
            chunk_id = f"{doc['doc_id']}#chunk_{index:03d}"
            embedding_text = f"Title: {doc['title']}\n\n{text}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc["doc_id"],
                    "chunk_index": index,
                    "title": doc["title"],
                    "text": text,
                    "embedding_text": embedding_text,
                    "word_count": word_count(text),
                    "char_count": len(text),
                }
            )
    return chunks


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_embeddings(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    existing = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[row["chunk_id"]] = row["embedding"]
    return existing


def append_embeddings(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            f.flush()


def call_openrouter_embeddings(
    *,
    api_key: str,
    model: str,
    texts: list[str],
    max_retries: int,
) -> tuple[list[list[float]], dict[str, Any] | None]:
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        request = Request(
            OPENROUTER_EMBEDDINGS_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "tau3-banking-kb-index",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {raw[:1000]}") from exc
            sleep_for = min(90, (2**attempt) + random.random())
            print(
                f"rate/server limit HTTP {exc.code}; retry {attempt + 1}/{max_retries} "
                f"after {sleep_for:.1f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            continue
        except URLError as exc:
            sleep_for = min(60, (2**attempt) + random.random())
            print(
                f"connection error; retry {attempt + 1}/{max_retries} after {sleep_for:.1f}s: {exc}",
                flush=True,
            )
            time.sleep(sleep_for)
            continue

        if "error" in data:
            message = json.dumps(data["error"])[:1000]
            code = data["error"].get("code")
            if code in {408, 409, 425, 429, 500, 502, 503, 504}:
                sleep_for = min(90, (2**attempt) + random.random())
                print(
                    f"rate/server limit API {code}; retry {attempt + 1}/{max_retries} "
                    f"after {sleep_for:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_for)
                continue
            raise RuntimeError(f"OpenRouter API error: {message}")

        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings, data.get("usage")

    raise RuntimeError("OpenRouter embedding retries exhausted, likely rate limited.")


def consolidate_embeddings(
    chunks: list[dict[str, Any]], embeddings_by_id: dict[str, list[float]], out_dir: Path
) -> None:
    missing = [chunk["chunk_id"] for chunk in chunks if chunk["chunk_id"] not in embeddings_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} embeddings, first missing: {missing[0]}")

    matrix = np.array([embeddings_by_id[chunk["chunk_id"]] for chunk in chunks], dtype=np.float32)
    np.save(out_dir / "embeddings.npy", matrix)
    (out_dir / "chunk_ids.json").write_text(
        json.dumps([chunk["chunk_id"] for chunk in chunks], indent=2)
    )


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)

    docs = load_documents(args.docs_dir)
    chunks = build_chunks(docs, args.max_words, args.overlap_words)

    docs_manifest = [
        {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "source_file": doc["source_file"],
            "content_sha256": sha256_text(doc["content"]),
            "word_count": word_count(doc["content"]),
            "char_count": len(doc["content"]),
        }
        for doc in docs
    ]
    chunk_manifest = [
        {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "chunk_index": chunk["chunk_index"],
            "title": chunk["title"],
            "text": chunk["text"],
            "embedding_text_sha256": sha256_text(chunk["embedding_text"]),
            "word_count": chunk["word_count"],
            "char_count": chunk["char_count"],
        }
        for chunk in chunks
    ]

    write_jsonl(args.out_dir / "docs.jsonl", docs_manifest)
    write_jsonl(args.out_dir / "chunks.jsonl", chunk_manifest)

    embeddings_path = args.out_dir / "embeddings.jsonl"
    embeddings_by_id = load_existing_embeddings(embeddings_path)

    pending = [chunk for chunk in chunks if chunk["chunk_id"] not in embeddings_by_id]
    print(
        f"docs={len(docs)} chunks={len(chunks)} already_embedded={len(embeddings_by_id)} "
        f"pending={len(pending)} out={args.out_dir}",
        flush=True,
    )

    total_prompt_tokens = 0
    total_cost = 0.0
    for start in range(0, len(pending), args.batch_size):
        batch = pending[start : start + args.batch_size]
        texts = [chunk["embedding_text"] for chunk in batch]
        embeddings, usage = call_openrouter_embeddings(
            api_key=api_key,
            model=args.model,
            texts=texts,
            max_retries=args.max_retries,
        )
        rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
            }
            for chunk, embedding in zip(batch, embeddings)
        ]
        append_embeddings(embeddings_path, rows)
        for row in rows:
            embeddings_by_id[row["chunk_id"]] = row["embedding"]

        if usage:
            total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            total_cost += float(usage.get("cost") or 0.0)

        done = len(embeddings_by_id)
        print(
            f"embedded {done}/{len(chunks)}"
            + (
                f" tokens+={usage.get('prompt_tokens')} cost+={usage.get('cost')}"
                if usage
                else ""
            ),
            flush=True,
        )
        if args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    consolidate_embeddings(chunks, embeddings_by_id, args.out_dir)

    matrix = np.load(args.out_dir / "embeddings.npy", mmap_mode="r")
    manifest = {
        "model": args.model,
        "embedding_dimensions": int(matrix.shape[1]),
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "max_words": args.max_words,
        "overlap_words": args.overlap_words,
        "docs_dir": str(args.docs_dir.relative_to(REPO_ROOT)),
        "artifact_files": [
            "manifest.json",
            "docs.jsonl",
            "chunks.jsonl",
            "embeddings.jsonl",
            "embeddings.npy",
            "chunk_ids.json",
        ],
        "run_usage": {
            "prompt_tokens": total_prompt_tokens,
            "cost": total_cost,
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
