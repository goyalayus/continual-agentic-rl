"""Hybrid BM25 + embedding retrieval over the banking knowledge documents."""

from __future__ import annotations

import json
import math
import os
import re
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = (
    REPO_ROOT
    / "data"
    / ".embeddings_cache"
    / "banking_knowledge"
    / "custom_chunk_index_qwen3_8b"
)
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
TOKEN_RE = re.compile(r"[a-z0-9_]+")
QUERY_ALIASES = {
    "mailed": ["mail", "offer", "promotion", "promotional"],
    "mail": ["mailed", "offer", "promotion", "promotional"],
    "offer": ["promotion", "promotional", "preapproved", "mailed"],
    "offers": ["promotion", "promotional", "preapproved", "mailed"],
    "expired": ["unavailable", "ineligible", "promotion", "offer"],
    "unavailable": ["expired", "ineligible", "promotion", "offer"],
    "cashback": ["cash", "back", "rewards"],
    "cash": ["cashback", "rewards"],
    "rewards": ["cashback", "points"],
    "stolen": ["lost", "replacement", "replace"],
    "lost": ["stolen", "replacement", "replace"],
    "cancel": ["close", "closure", "deactivate"],
    "close": ["cancel", "closure", "deactivate"],
}


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    title: str
    summary: str


class SimpleBM25:
    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_docs = tokenized_docs
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_docs)
        self.doc_lengths = [len(doc) for doc in tokenized_docs]
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.doc_count if self.doc_count else 0.0
        )
        self.term_freqs = [Counter(doc) for doc in tokenized_docs]
        doc_freqs: dict[str, int] = defaultdict(int)
        for doc in tokenized_docs:
            for term in set(doc):
                doc_freqs[term] += 1
        self.idf = {
            term: math.log(1 + (self.doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freqs.items()
        }

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.doc_count, dtype=np.float32)
        if not query_tokens or not self.doc_count:
            return scores
        query_terms = set(query_tokens)
        for index, freqs in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[index] or 1
            denom_base = self.k1 * (
                1 - self.b + self.b * doc_len / (self.avg_doc_length or 1)
            )
            score = 0.0
            for term in query_terms:
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                score += idf * (freq * (self.k1 + 1)) / (freq + denom_base)
            scores[index] = score
        return scores


class BankingHybridRetriever:
    """Searches banking KB chunks, then returns document-level summary hits."""

    def __init__(
        self,
        index_dir: Path | str = DEFAULT_INDEX_DIR,
        *,
        embedding_model: str = "qwen/qwen3-embedding-8b",
        embedding_weight: float = 0.55,
        bm25_weight: float = 0.45,
        request_timeout_seconds: int = 60,
        event_logger: Callable[..., None] | None = None,
    ):
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model
        self.embedding_weight = embedding_weight
        self.bm25_weight = bm25_weight
        self.request_timeout_seconds = request_timeout_seconds
        self.event_logger = event_logger
        self._embedding_disabled_logged = False

        self.docs = self._load_jsonl("docs.jsonl")
        self.chunks = self._load_jsonl("chunks.jsonl")
        self.manifest = json.loads((self.index_dir / "manifest.json").read_text())
        self.chunk_ids = json.loads((self.index_dir / "chunk_ids.json").read_text())
        self.summaries = json.loads(
            (self.index_dir / "summaries_by_doc_id.json").read_text()
        )
        self.embeddings = np.load(self.index_dir / "embeddings.npy")

        self.docs_by_id = {doc["doc_id"]: doc for doc in self.docs}
        self.chunks_by_id = {chunk["chunk_id"]: chunk for chunk in self.chunks}
        self._validate_artifacts()
        self._normalize_embeddings()
        self.chunk_by_position = [
            self.chunks_by_id[chunk_id] for chunk_id in self.chunk_ids
        ]
        self._bm25 = SimpleBM25(
            [
                self._tokenize(self._chunk_search_text(chunk))
                for chunk in self.chunk_by_position
            ]
        )
        self._query_embedding_cache: dict[str, np.ndarray | None] = {}

    def search(self, query: str, top_k: int = 10) -> list[SearchHit]:
        query = query.strip()
        if not query:
            return []
        top_k = self._clamp_top_k(top_k)

        expanded_query = self._expand_query(query)
        bm25_scores = self._bm25.score(self._tokenize(expanded_query))
        embedding_scores = self._embedding_scores(query)
        combined = self._combine_scores(bm25_scores, embedding_scores)

        ranked_chunk_indices = np.argsort(combined)[::-1]
        seen_doc_ids: set[str] = set()
        hits: list[SearchHit] = []
        for chunk_index in ranked_chunk_indices:
            if combined[chunk_index] <= 0:
                break
            chunk = self.chunk_by_position[int(chunk_index)]
            doc_id = chunk["doc_id"]
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)
            hits.append(self._hit_for_doc(doc_id))
            if len(hits) >= top_k:
                break

        return hits

    def read_doc(self, doc_id: str) -> str:
        doc_id = doc_id.strip()
        if doc_id not in self.docs_by_id:
            close = self._closest_doc_ids(doc_id)
            suffix = f" Did you mean: {', '.join(close)}?" if close else ""
            raise KeyError(f"Unknown document id: {doc_id}.{suffix}")

        source_file = REPO_ROOT / self.docs_by_id[doc_id]["source_file"]
        raw = json.loads(source_file.read_text())
        return f"# {raw['title']}\n\n{raw['content']}".strip()

    def format_search_results(self, hits: list[SearchHit]) -> str:
        if not hits:
            return "No matching knowledge documents found."

        blocks = []
        for index, hit in enumerate(hits, start=1):
            blocks.append(
                "\n".join(
                    [
                        f"{index}. {hit.doc_id}",
                        f"Title: {hit.title}",
                        f"Summary: {hit.summary}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        path = self.index_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing KB index artifact: {path}")
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def _validate_artifacts(self) -> None:
        missing = []
        if not self.docs:
            missing.append("docs.jsonl has no rows")
        if not self.chunks:
            missing.append("chunks.jsonl has no rows")
        if self.manifest.get("model") != self.embedding_model:
            missing.append(
                "manifest model does not match retriever embedding_model "
                f"({self.manifest.get('model')} != {self.embedding_model})"
            )
        if self.embeddings.ndim != 2:
            missing.append(
                f"embeddings.npy must be 2D, got shape {self.embeddings.shape}"
            )
        elif self.manifest.get("embedding_dimensions") != int(self.embeddings.shape[1]):
            missing.append(
                "manifest embedding_dimensions does not match embeddings.npy "
                f"({self.manifest.get('embedding_dimensions')} != {self.embeddings.shape[1]})"
            )
        if len(self.docs_by_id) != len(self.docs):
            missing.append("docs.jsonl contains duplicate doc_id values")
        if len(self.chunks_by_id) != len(self.chunks):
            missing.append("chunks.jsonl contains duplicate chunk_id values")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            missing.append("chunk_ids.json contains duplicate chunk ids")
        embedding_rows = int(self.embeddings.shape[0]) if self.embeddings.ndim >= 1 else 0
        if len(self.chunk_ids) != embedding_rows:
            missing.append(
                "chunk_ids.json length does not match embeddings.npy rows "
                f"({len(self.chunk_ids)} != {embedding_rows})"
            )
        for chunk_id in self.chunk_ids:
            if chunk_id not in self.chunks_by_id:
                missing.append(f"chunk id missing from chunks.jsonl: {chunk_id}")
                break
        for doc in self.docs:
            doc_id = doc.get("doc_id")
            if doc_id not in self.summaries:
                missing.append(f"summary missing for doc_id: {doc_id}")
                break
            source_file = REPO_ROOT / doc.get("source_file", "")
            if not source_file.exists():
                missing.append(f"source document missing for doc_id {doc_id}: {source_file}")
                break
            try:
                source = json.loads(source_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                missing.append(f"source document unreadable for doc_id {doc_id}: {exc}")
                break
            content_hash = self._sha256(source.get("content", ""))
            if doc.get("content_sha256") != content_hash:
                missing.append(
                    f"source content hash mismatch for doc_id {doc_id}: "
                    f"{doc.get('content_sha256')} != {content_hash}"
                )
                break
        for chunk in self.chunks:
            if chunk.get("doc_id") not in self.docs_by_id:
                missing.append(f"chunk references unknown doc_id: {chunk.get('doc_id')}")
                break
            embedding_text = f"Title: {chunk.get('title', '')}\n\n{chunk.get('text', '')}"
            text_hash = self._sha256(embedding_text)
            if chunk.get("embedding_text_sha256") != text_hash:
                missing.append(
                    f"chunk text hash mismatch for chunk_id {chunk.get('chunk_id')}: "
                    f"{chunk.get('embedding_text_sha256')} != {text_hash}"
                )
                break
        if missing:
            raise ValueError(
                "Invalid banking KB index artifacts in "
                f"{self.index_dir}:\n- " + "\n- ".join(missing)
            )

    def _normalize_embeddings(self) -> None:
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.embeddings = self.embeddings / norms

    def _hit_for_doc(self, doc_id: str) -> SearchHit:
        summary_row = self.summaries[doc_id]
        return SearchHit(
            doc_id=doc_id,
            title=summary_row["title"],
            summary=summary_row["summary"],
        )

    def _embedding_scores(self, query: str) -> np.ndarray | None:
        embedding = self._query_embedding(query)
        if embedding is None:
            return None
        if embedding.shape[0] != self.embeddings.shape[1]:
            self._log_event(
                "embedding_error",
                reason="dimension_mismatch",
                query=query,
                expected=self.embeddings.shape[1],
                received=embedding.shape[0],
            )
            return None
        try:
            return self.embeddings @ embedding
        except Exception as exc:
            self._log_event("embedding_error", reason="matmul_failed", error=str(exc))
            return None

    def _query_embedding(self, query: str) -> np.ndarray | None:
        if query in self._query_embedding_cache:
            return self._query_embedding_cache[query]

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            if not self._embedding_disabled_logged:
                self._log_event(
                    "embedding_disabled",
                    reason="OPENROUTER_API_KEY is not set; using BM25 only",
                )
                self._embedding_disabled_logged = True
            self._query_embedding_cache[query] = None
            return None

        payload = {
            "model": self.embedding_model,
            "input": [
                "Instruct: Retrieve banking policy passages that answer the query.\n"
                f"Query: {query}"
            ],
            "encoding_format": "float",
        }
        request = Request(
            OPENROUTER_EMBEDDINGS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost",
                "X-Title": "tau3-custom-harness",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
            embedding = np.array(data["data"][0]["embedding"], dtype=np.float32)
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            self._log_event(
                "embedding_error",
                reason="query_embedding_failed",
                query=query,
                error=str(exc),
            )
            self._query_embedding_cache[query] = None
            return None

        norm = np.linalg.norm(embedding)
        if norm:
            embedding = embedding / norm
        self._query_embedding_cache[query] = embedding
        return embedding

    def _combine_scores(
        self, bm25_scores: np.ndarray, embedding_scores: np.ndarray | None
    ) -> np.ndarray:
        bm25_norm = self._minmax(bm25_scores)
        if embedding_scores is None:
            return bm25_norm
        embedding_norm = self._minmax(embedding_scores)
        return self.bm25_weight * bm25_norm + self.embedding_weight * embedding_norm

    def _minmax(self, scores: np.ndarray) -> np.ndarray:
        if scores.size == 0:
            return scores
        min_score = float(np.min(scores))
        max_score = float(np.max(scores))
        if math.isclose(min_score, max_score):
            return np.zeros_like(scores, dtype=np.float32)
        return ((scores - min_score) / (max_score - min_score)).astype(np.float32)

    def _closest_doc_ids(self, value: str) -> list[str]:
        value_tokens = set(self._tokenize(value))
        if not value_tokens:
            return []
        scored = []
        for doc_id in self.docs_by_id:
            tokens = set(self._tokenize(doc_id))
            overlap = len(value_tokens & tokens)
            if overlap:
                scored.append((overlap, doc_id))
        scored.sort(reverse=True)
        return [doc_id for _, doc_id in scored[:3]]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_RE.findall(text.lower())

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _chunk_search_text(self, chunk: dict[str, Any]) -> str:
        doc_id = chunk["doc_id"]
        summary = self.summaries.get(doc_id, {}).get("summary", "")
        return f"{chunk.get('title', '')} {summary} {chunk.get('text', '')}"

    def _expand_query(self, query: str) -> str:
        tokens = self._tokenize(query)
        aliases = []
        for token in tokens:
            aliases.extend(QUERY_ALIASES.get(token, []))
        if not aliases:
            return query
        return query + " " + " ".join(aliases)

    def _clamp_top_k(self, top_k: int) -> int:
        try:
            value = int(top_k)
        except (TypeError, ValueError):
            value = 10
        return max(1, min(value, 20))

    def _log_event(self, event_type: str, **payload: Any) -> None:
        if self.event_logger is not None:
            self.event_logger(event_type, **payload)
