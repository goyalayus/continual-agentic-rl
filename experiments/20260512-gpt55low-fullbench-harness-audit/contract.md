# GPT-5.5 Low Banking Harness Audit

## Objective

Run the banking benchmark with two harnesses, compare failures task-by-task, then improve only the custom harness prompts/tool descriptions and rerun the custom harness.

## Scope

- Domain: `banking_knowledge`
- Task set: all 97 banking tasks
- Repeats: 4 runs per task
- Baseline A: current custom batch-subagent harness
- Baseline B: default Tau harness
- Repair surface: prompts and tool descriptions only
- No prompt leaks: fixes must be general behavioral or tool-contract guidance, never task-specific answers, task ids, exact expected actions, or hidden benchmark labels

## Model Contract

- Intended agent model: GPT-5.5 low reasoning
- Intended user model: GPT-5.5 low reasoning
- Intended custom subagent model: GPT-5.5 low reasoning
- Azure deployment note: `azure/gpt-5.5` has enough quota but rejected some banking prompts through provider safety filtering, so it is not a valid provider for the full banking comparison unless that filter behavior is fixed.
- OpenRouter note: `openrouter/openai/gpt-5.5` is the preferred provider path for this experiment because LiteLLM accepts it and returns reasoning details. The run is currently blocked until a valid OpenRouter key is supplied through `OPENROUTER_API_KEY` or gitignored `.env.local`.
- Reasoning: low
- Temperature: `1.0`
- Max output tokens per LLM call: `768`
- Shared base seed: `4101`
- Trial seeds matching Tau's seed generation from `4101`: `849558`, `551167`, `811445`, `613921`

## Retrieval Contract

- Custom harness: its current hybrid knowledge retrieval, BM25 plus query embeddings, unless provider embedding preflight fails.
- Default Tau harness: explicit `--retrieval-config bm25`.
- If custom embedding preflight fails, do not silently run the main benchmark as if hybrid worked. Either restore embeddings or mark the run as a degraded-control run.

## Fairness Rules

- Same model family and reasoning level for agent and user in both harnesses.
- Same task set and four-repeat count.
- Same max steps, max errors, timeout, temperature, and max output tokens where the harness supports them.
- Do not change prompts or tool descriptions before baseline logs are collected.
- Do not use the failure table to write task-specific instructions.

## Artifacts

- Baseline custom artifacts:
  - one custom single-file artifact per seed, produced by `experiments/20260509-gpt54mini-harness/run_azure_batch.py`
  - underlying run folders under `experiments/20260509-gpt54mini-harness/runs/`
- Baseline default artifact:
  - Tau `results.json` under `data/simulations/<run-name>/`
  - verbose per-simulation artifacts if enabled
- Launch-state artifacts:
  - `baseline_launch_state.json` before the baseline provider preflight
  - `postfix_launch_state.json` before the post-fix provider preflight
  - these record branch, commit, dirty file names, and untracked file names, but not diff contents or secrets
- Failure audit table:
  - `experiments/20260512-gpt55low-fullbench-harness-audit/failure_audit.md`
  - machine-readable sidecar `failure_audit.json`
- Fix branch:
  - created only after the failure audit exists
  - contains prompt/tool-description-only changes
- Post-fix custom benchmark artifact:
  - four-repeat custom benchmark after fixes
  - final comparison summary

## Required Analysis Columns

For each task:

- task id
- custom pass count out of 4
- default pass count out of 4
- custom failed run ids
- expected actions and reward basis
- observed custom actions/messages in failed runs
- exact custom failure mode
- what I think is wrong in the harness behavior
- prompt/tool-description-only fix idea
- anti-cheating check

## Stopping Rules

- If provider credits or quota fail mid-run, stop and preserve completed artifacts. Resume with the same run names after credits are restored.
- If a runner bug invalidates logs, fix the runner and rerun only invalid/missing runs.
- If the failure audit cannot be generated from available logs, do not make prompt changes yet.
- If any proposed fix requires code beyond prompts/tool descriptions, record it but do not implement it in the fix branch.

## Verification Gates

- Preflight custom runner on one task before full custom run.
- Preflight default Tau runner on one task before full default run.
- Confirm every baseline task has exactly four scored attempts per harness before writing the audit.
- Confirm the fix branch diff only touches prompt/tool-description files.
- Rerun smoke tests for prompt/tool wiring after edits.
- Confirm post-fix custom benchmark has four scored attempts per task.
