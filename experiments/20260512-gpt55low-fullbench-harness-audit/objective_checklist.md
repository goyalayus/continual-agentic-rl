# Objective Checklist

This file maps the active objective to concrete artifacts and gates. Do not
call the objective complete from memory or intent; use this checklist and
`completion_audit.py`.

| Requirement | Evidence / Gate | Current Status |
| --- | --- | --- |
| Provider can run the frozen GPT-5.5 low contract | `uv run python experiments/20260512-gpt55low-fullbench-harness-audit/provider_preflight.py` prints `provider_preflight_ok` after `OPENROUTER_API_KEY` is available from the current environment or gitignored `.env.local`; this checks GPT-5.5 low chat plus the custom hybrid-retriever path | Blocked as of 2026-05-12 12:53 IST: no `OPENROUTER_API_KEY`, no `.env.local`, watcher is installed as an enabled user service with linger enabled, and the contract still requires `max_tokens=768` |
| Baseline launch worktree state is recorded | `run_full_baselines_openrouter.sh` writes `baseline_launch_state.json` before provider preflight; `completion_audit.py` requires the label, commit, and status list | Missing because the baseline has not launched |
| Baseline custom batch-subagent harness runs all 97 tasks 4 times | `comparison_summary.json` has exactly the task IDs from `data/tau2/domains/banking_knowledge/tasks`, no duplicate task rows, and 388 scored custom attempts; `check_completeness.py` passes | Missing: status currently shows `custom scored: 0/388` |
| Baseline default Tau harness runs all 97 tasks 4 times | `comparison_summary.json` has exactly the task IDs from `data/tau2/domains/banking_knowledge/tasks`, no duplicate task rows, and 388 scored default attempts; `check_completeness.py` passes | Missing: status currently shows `default scored: 0/388` |
| Baseline artifacts match the frozen run contract | `check_completeness.py` validates exact task IDs, duplicate-free task rows, exact per-task run counts, unique run IDs, source-label counts, model, reasoning effort, `max_tokens`, temperature, seeds, retrieval mode, max steps/errors, and harness identity | Not satisfied because no valid baseline runs exist |
| Per-task audit packets exist for all 97 tasks | `generate_failure_packets.py` writes `failure_packets/manifest.json` and 97 complete Markdown packets | Missing; only incomplete dry-run packets may exist |
| Codex reviews each task one by one and writes what custom did wrong | Every packet has filled `Exact custom failure mode`, `Suspected harness behavior`, `General prompt/tool-description fix idea`, `Anti-cheating check`, `Codex review confirmation`, and `Reviewer verdict`; compiler/final audit require the confirmation to explicitly say Codex reviewed the task | Missing |
| Final per-task failure audit table exists | `compile_failure_audit.py` writes `failure_audit.md` and `failure_audit.json`; compiler enforces 97 unique packets and non-vague diagnosis notes | Missing |
| Generic prompt-fix handoff exists without task leaks | `compile_failure_audit.py` writes `generic_fix_input.md` and `generic_fix_input.json`; `completion_audit.py` validates schema, forbidden markers, `task_042`/`task 042` patterns, and exact blocked task-data terms from `failure_audit.json` | Missing |
| Prompt/tool-description fix branch exists | Current branch is non-main after the baseline audit is complete | Missing; current branch is `main` |
| Fix branch changes only prompts/tool descriptions | `check_prompt_only_diff.py --base main` passes across committed branch changes, staged changes, unstaged working-tree changes, and relevant untracked prompt/tool files; it uses `failure_audit.json` to reject exact leaks while requiring safe `generic_fix_input.json` | Missing |
| Prompt changes do not cheat or prompt-leak | Manual review plus `check_prompt_only_diff.py`; no task ids, expected actions, hidden labels, customer data, or exact benchmark arguments in added prompt/tool-description text | Missing |
| Post-fix custom harness reruns all 97 tasks 4 times | `run_postfix_custom_openrouter.sh` writes `postfix_comparison_summary.json/.csv`; custom-only `check_completeness.py` passes with prefix `postfix_custom_openrouter_gpt55low_` | Missing |
| Post-fix launch worktree state is recorded | `run_postfix_custom_openrouter.sh` writes `postfix_launch_state.json` before provider preflight; `completion_audit.py` requires the label, commit, and status list | Missing because the post-fix run has not launched |
| Post-fix accuracy is measured | `completion_audit.py` reports 97 tasks and 388 scored post-fix custom attempts, with pass count | Missing |
| Full objective is complete | `python3 experiments/20260512-gpt55low-fullbench-harness-audit/completion_audit.py` prints `completion_audit: complete` | Not complete |

## Next Valid Action

Supply a valid OpenRouter key through the gitignored helper. This is the
least-racy path while `tau2-openrouter-key-watch.service` is active:

```bash
cd /home/ayush/tau2-bench
experiments/20260512-gpt55low-fullbench-harness-audit/setup_openrouter_env.sh
```

The helper prompts without echo, writes:

```bash
OPENROUTER_API_KEY=sk-or-...
```

to `experiments/20260512-gpt55low-fullbench-harness-audit/.env.local` with
mode `600`, then restarts the watcher so the provider preflight runs promptly.
The scripts read only `OPENROUTER_API_KEY` from that file.

To prompt for the key and immediately start the baseline:

```bash
experiments/20260512-gpt55low-fullbench-harness-audit/setup_and_run_full_baselines_openrouter.sh
```

The direct setup-and-run wrapper stops the watcher after the key prompt succeeds,
then launches the baseline itself.

To wait until the key and provider are ready before starting the baseline:

```bash
INTERVAL_SECONDS=60 \
  experiments/20260512-gpt55low-fullbench-harness-audit/wait_for_key_and_provider_and_run.sh
```

To keep that watcher alive under user systemd:

```bash
experiments/20260512-gpt55low-fullbench-harness-audit/start_key_provider_watcher.sh
```

The systemd watcher helper defaults to a 60-second poll interval unless
`INTERVAL_SECONDS` is set. If that service is already running, use
`setup_openrouter_env.sh` so the key is written to `.env.local` and the watcher
is restarted promptly; exporting the key in a separate shell will not update an
already-running service process.

For a watcher that is enabled as a normal user service across later user
sessions, run:

```bash
experiments/20260512-gpt55low-fullbench-harness-audit/install_key_provider_watcher_service.sh
```

Do not lower `max_tokens` or change the model contract just to fit the current
credit balance. That would create a different experiment.
