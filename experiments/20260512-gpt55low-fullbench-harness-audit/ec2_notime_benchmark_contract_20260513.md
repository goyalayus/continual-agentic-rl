# EC2 No-Time Benchmark Contract - 2026-05-13

## Objective

Rerun the full banking benchmark on EC2 after removing the per-conversation wall-clock timeout and adding a generic custom-agent scope-discipline prompt.

## Run Shape

- Domain: `banking_knowledge`
- Tasks: all 97 banking tasks
- Trials: 4 per task for both harnesses
- Harnesses: custom planner/subagent harness and default Tau BM25 harness
- Launcher: `experiments/20260512-gpt55low-fullbench-harness-audit/run_full_baselines_one_process.py`
- Parallelism: `CUSTOM_PARALLELISM=388` and default `--max-concurrency 388`
- Wall-clock simulation timeout: disabled (`timeout_seconds=None`)
- Max steps: unchanged at 100
- Resume behavior: custom reruns failed/missing attempts with `--auto-resume`; transient provider transport/server errors may be retried twice inside a resumed attempt, but auth, quota, content-policy, and validation errors are not retried by that wrapper.
- LLM-call retry fix: new/resumed Python processes retry transient Azure Responses API transport errors at the individual call layer and use a larger LiteLLM HTTP connection pool. The already-running first pass does not get this patch retroactively.

## Model Contract

- Chat model: `azure/gpt-5.5`
- Reasoning effort: `low`
- Max output tokens: `768`
- Temperature: `1.0`
- Embeddings: OpenRouter `qwen/qwen3-embedding-8b` for custom retrieval query embeddings
- Credentials must load from the experiment `.env.local` flow.

## Prompt Change

The custom agent gets a generic scope-discipline instruction: solve the customer request, avoid optional adjacent banking actions, act once policy evidence is enough, and stop after confirming completed work.

## Artifacts

- Run log prefix: `ec2_notime_benchmark_20260513`
- Custom prefix: `notime_custom_azure_gpt55low_`
- Default save directory: `notime_default_tau_bm25_azure_gpt55low_4trials_seed4101`
- Analysis JSON: `experiments/20260512-gpt55low-fullbench-harness-audit/notime_comparison_summary.json`
- Analysis CSV: `experiments/20260512-gpt55low-fullbench-harness-audit/notime_comparison_summary.csv`

## Completion Criteria

- Provider preflight passes on EC2.
- Both harnesses complete all 388 attempts, or a provider/resource failure is preserved with logs.
- Final report includes pass counts, pass rates, makespan, throughput, and termination/noise counts.

## Stop Rules

- Stop if provider credits, auth, or quota fail in a way that makes continued calls wasteful.
- Stop if EC2 memory pressure approaches instability.
- Preserve partial artifacts before cleanup.
