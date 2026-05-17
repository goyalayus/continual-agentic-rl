# Tau3 Custom Harness

This folder is the custom banking harness we discussed.

The shape is:

- Tau still owns the banking environment, user simulator, DB tools, task loading,
  and evaluator.
- The environment is built with `retrieval_variant="no_knowledge"`, so Tau's old
  KB search, grep, and shell tools are not exposed.
- The planner sees the banking DB/action tools plus one internal tool:
  `ask_knowledge_subagents(requests)`. It accepts 1 to 4 labeled KB research
  requests, runs those subagents in parallel, and returns labeled notes.
- The runner uses `SafeUserSimulator`, a thin wrapper around Tau's user
  simulator that retries empty model outputs and falls back to `###STOP###`
  instead of crashing the whole run.
- KB subagents see only:
  `search(query, top_k=10)` and `read_doc(doc_id)`.
  `query` can be one string or up to 3 related query strings. Multi-query
  search merges and dedupes results before returning summaries.
- The planner talks to the user. KB subagents never talk to the user and never
  call DB tools.
- Subagents cannot spawn subagents. The only allowed shape is
  planner -> subagent, so the planner must break KB work into separate subagent
  questions itself.
- Discoverable agent/user tools are evidence-bound in the harness: the planner
  must have a KB document read that contains the exact hidden tool name before
  it can unlock, call, or give that tool.
- If a KB lookup fails because the provider errors or the prompt is too large,
  the planner is blocked from state-changing banking tools until it successfully
  retries policy lookup, asks for safe clarification, or transfers.

The KB search is hybrid:

- BM25 over all document chunks.
- Qwen3 embedding search over the precomputed chunk matrix when
  `OPENROUTER_API_KEY` is set.
- Results are grouped back to documents.
- The model only sees `doc_id`, `title`, and `summary`.
- Full policy text is available only through `read_doc(doc_id)`.

Run one task:

```bash
OPENROUTER_API_KEY=... \
TAU3_AGENT_MODEL=gpt-4.1 \
TAU3_USER_MODEL=gpt-4.1 \
uv run python custom_harness/tau3_custom_harness/run_banking.py --task-id task_001
```

Run without the evaluator:

```bash
uv run python custom_harness/tau3_custom_harness/run_banking.py --task-id task_001 --skip-eval
```

Smoke test the wiring without calling an LLM:

```bash
uv run python custom_harness/tau3_custom_harness/smoke_test.py
```

Logs go to `benchmark_evaluation/custom_harness_runs/<run_id>/` as:

- `events.jsonl`: planner/subagent/search/read events
- `simulation.json`: Tau simulation object with messages and reward info
- `kb_evidence.json`: KB docs read by subagents and exact hidden tool names seen
- `llm_calls/`: raw LLM request/response logs when a live model run is used

This is local-first on purpose. The JSONL folder can be uploaded to S3 later
without changing the trace format.

To upload at the end of a run:

```bash
uv run python custom_harness/tau3_custom_harness/run_banking.py --task-id task_001 --s3-uri s3://your-bucket/tau3-runs
```

That uses `aws s3 sync`. Upload failures are logged but do not fail the local run
unless you pass `--s3-strict`.
