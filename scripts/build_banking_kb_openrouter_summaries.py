#!/usr/bin/env python3
"""Generate one-document-at-a-time KB summaries with OpenRouter."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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
OPENROUTER_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


SYSTEM_PROMPT = """You summarize Rho-Bank banking knowledge-base documents for search results.

The summary is shown to an agent after retrieval. Its job is only to help the agent decide whether to read the full source document.

Rules:
- Summarize exactly one document.
- Write one sentence, ideally 25-55 words.
- Mention concrete operations, eligibility, limits, exceptions, tool names, or customer facts when present.
- Do not say "this document", "this guide", "KB", or "article".
- Do not invent facts.
- Preserve exact tool names, dollar amounts, percentages, dates, account/card names, and odd source values.
- Return valid JSON only with this shape: {"summary": "..."}

Few-shot examples:

Title: Internal: Opening Personal Checking Accounts
Content:
## Eligibility Requirements
To open a personal checking account, ensure the customer is verified, at least 18, has no more than 4 personal checking accounts, and has no checking accounts closed for cause in the past 6 months.
## Opening Procedure
Verify identity, check eligibility, confirm the full official account_class name, then use open_bank_account_4821.
Output:
{"summary":"Covers opening a personal checking account, including identity verification, age and account-count eligibility, recent closure restrictions, official account_class naming, and use of open_bank_account_4821."}

Title: Internal: Filing a Debit Card Transaction Dispute
Content:
Customer must be verified, the transaction must be at least $1.00 and within 60 days, dispute limits vary by checking-account tier, and the debit card must be linked to an open checking account. Use file_debit_card_transaction_dispute_6281 with transaction, account, card, user, and dispute-category details.
Output:
{"summary":"Covers debit-card transaction disputes, including verification, Regulation E timing and liability, transaction age and amount rules, account-tier dispute limits, required tool arguments, and file_debit_card_transaction_dispute_6281."}

Title: Sending Limits: Complete Guide
Content:
New accounts can send $750 daily and $7,500 monthly. Verified accounts can send $3,500 daily and $35,000 monthly. Both daily and monthly limits apply at the same time; customers may request higher limits from account settings with up to 4 days review.
Output:
{"summary":"Explains Everyone Pay sending limits for new and verified accounts, how daily and monthly limits interact, where customers check remaining limits, and how to request a higher limit."}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--model", default="openai/gpt-5.4-mini")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=192)
    return parser.parse_args()


def load_documents(docs_dir: Path) -> list[dict[str, Any]]:
    docs = []
    for path in sorted(docs_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        try:
            source_file = str(path.relative_to(REPO_ROOT))
        except ValueError:
            source_file = str(path)
        docs.append(
            {
                "doc_id": raw["id"],
                "title": raw["title"],
                "content": raw["content"],
                "source_file": source_file,
            }
        )
    if not docs:
        raise RuntimeError(f"No document JSON files found in {docs_dir}")
    return docs


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def load_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[row["doc_id"]] = row
    return rows


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match is None:
            raise
        return json.loads(match.group(0))


def normalize_summary(summary: str) -> str:
    summary = " ".join(summary.split())
    if len(summary) > 800:
        summary = summary[:797].rstrip() + "..."
    return summary


def call_openrouter_summary(
    *,
    api_key: str,
    model: str,
    doc: dict[str, Any],
    max_retries: int,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Document ID: {doc['doc_id']}\n"
                f"Title: {doc['title']}\n\n"
                f"Content:\n{doc['content']}\n\n"
                "Return JSON only."
            ),
        },
    ]
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    body = json.dumps(payload).encode("utf-8")

    for attempt in range(max_retries):
        request = Request(
            OPENROUTER_CHAT_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "tau3-banking-kb-summaries",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if not retryable:
                raise RuntimeError(f"OpenRouter HTTP {exc.code}: {raw[:1000]}") from exc
            sleep_for = min(120, (2**attempt) + random.random())
            print(
                f"{doc['doc_id']}: HTTP {exc.code}; retry {attempt + 1}/{max_retries} "
                f"after {sleep_for:.1f}s",
                flush=True,
            )
            time.sleep(sleep_for)
            continue
        except URLError as exc:
            sleep_for = min(90, (2**attempt) + random.random())
            print(
                f"{doc['doc_id']}: connection error; retry {attempt + 1}/{max_retries} "
                f"after {sleep_for:.1f}s: {exc}",
                flush=True,
            )
            time.sleep(sleep_for)
            continue

        if "error" in data:
            error = data["error"]
            code = error.get("code")
            if code in {408, 409, 425, 429, 500, 502, 503, 504}:
                sleep_for = min(120, (2**attempt) + random.random())
                print(
                    f"{doc['doc_id']}: API {code}; retry {attempt + 1}/{max_retries} "
                    f"after {sleep_for:.1f}s",
                    flush=True,
                )
                time.sleep(sleep_for)
                continue
            raise RuntimeError(f"OpenRouter API error: {json.dumps(error)[:1000]}")

        content = data["choices"][0]["message"]["content"]
        parsed = extract_json_object(content)
        summary = normalize_summary(parsed["summary"])
        row = {
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "summary": summary,
            "source_file": doc["source_file"],
            "content_sha256": sha256_text(doc["content"]),
            "content_word_count": word_count(doc["content"]),
            "model": model,
        }
        return row, data.get("usage")

    raise RuntimeError(f"{doc['doc_id']}: retries exhausted, likely rate limited.")


def append_row(path: Path, row: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        with path.open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def main() -> int:
    args = parse_args()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries_path = args.out_dir / "summaries.jsonl"
    docs = load_documents(args.docs_dir)
    existing = load_existing(summaries_path)
    pending = [doc for doc in docs if doc["doc_id"] not in existing]

    print(
        f"docs={len(docs)} already_summarized={len(existing)} pending={len(pending)} "
        f"model={args.model} out={summaries_path}",
        flush=True,
    )

    lock = threading.Lock()
    completed = len(existing)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost = 0.0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                call_openrouter_summary,
                api_key=api_key,
                model=args.model,
                doc=doc,
                max_retries=args.max_retries,
                max_tokens=args.max_tokens,
            ): doc
            for doc in pending
        }
        for future in concurrent.futures.as_completed(futures):
            doc = futures[future]
            try:
                row, usage = future.result()
            except Exception as exc:
                print(f"FAILED {doc['doc_id']}: {exc}", flush=True)
                raise
            append_row(summaries_path, row, lock)
            existing[row["doc_id"]] = row
            completed += 1
            if usage:
                total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                total_completion_tokens += int(usage.get("completion_tokens") or 0)
                total_cost += float(usage.get("cost") or 0.0)
            print(
                f"summarized {completed}/{len(docs)} {row['doc_id']}"
                + (f" cost+={usage.get('cost')}" if usage else ""),
                flush=True,
            )

    ordered = [existing[doc["doc_id"]] for doc in docs]
    (args.out_dir / "summaries_by_doc_id.json").write_text(
        json.dumps({row["doc_id"]: row for row in ordered}, indent=2, ensure_ascii=False)
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "num_documents": len(docs),
        "summary_file": "summaries.jsonl",
        "summary_map_file": "summaries_by_doc_id.json",
        "prompt_contract": {
            "one_document_per_call": True,
            "one_sentence": True,
            "word_target": "25-55",
            "json_only": True,
        },
        "run_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "cost": total_cost,
        },
    }
    (args.out_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
