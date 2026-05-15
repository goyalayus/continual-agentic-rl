# EC2 Full Benchmark Contract - 2026-05-13

## Objective

Run the full banking benchmark on a 16 GiB EC2 instance so the laptop does not carry the memory load.

## Run Shape

- Domain: `banking_knowledge`
- Task set: all 97 banking tasks
- Trials: 4 per task for both harnesses
- Harnesses: default Tau BM25 and custom planner/subagent harness
- Launcher: `experiments/20260512-gpt55low-fullbench-harness-audit/run_full_baselines_one_process.py`
- Parallelism: fully parallel for both sides unless the remote host shows memory pressure

## Model Contract

- Chat model: `azure/gpt-5.5`
- Reasoning effort: `low`
- Max output tokens: `768`
- Temperature: `1.0`
- Embeddings: OpenRouter `qwen/qwen3-embedding-8b` for custom retrieval query embeddings
- Credentials must come from the experiment `.env.local` flow, not transient shell-only exports.

## Artifacts

- Remote stdout/stderr log: `experiments/20260512-gpt55low-fullbench-harness-audit/ec2_full_benchmark_20260513.log`
- Remote resource samples: `experiments/20260512-gpt55low-fullbench-harness-audit/ec2_full_benchmark_20260513_resource.tsv`
- Custom artifacts: `experiments/20260509-gpt54mini-harness/artifacts/*.jsonl.gz`
- Default artifacts: `data/simulations/<default-run-name>/`
- Analysis output: `experiments/20260512-gpt55low-fullbench-harness-audit/comparison_summary.json`

## Completion Criteria

- Provider preflight passes on the EC2 instance.
- Both harnesses finish, or a provider/rate-limit failure is detected and preserved.
- Final counts are reported as completed attempts, pass counts, pass rates, makespan, and throughput.

## Stop Rules

- Stop if provider credits, authentication, or quota fail in a way that makes continued calls wasteful.
- Stop if memory pressure approaches EC2 instability.
- Preserve partial artifacts before any cleanup.
