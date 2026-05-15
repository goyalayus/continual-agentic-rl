# GPT-5.5 Low Banking Harness Audit

This folder is the control room for the full custom-vs-default banking run.
For the current prompt-to-artifact completion map, see `objective_checklist.md`.

## Current State

The benchmark is blocked on the local OpenRouter key, not harness code.

- Azure `gpt-5.5` has quota, but provider safety filters rejected some banking benchmark prompts.
- OpenRouter `openrouter/openai/gpt-5.5` is the selected provider path, but no usable `OPENROUTER_API_KEY` is currently visible from the shell or `.env.local`.
- `tau2-openrouter-key-watch.service` is installed as an enabled user service and waits for `.env.local`, then runs provider preflight before launching the full baseline.
- Existing outputs are partial and must not be treated as final benchmark scores.
- Generated summaries/logs may contain hidden expected actions, tool names,
  account ids, transaction ids, and customer PII. They are ignored by the local
  `.gitignore` and should stay local unless explicitly redacted.

## Resume Command

Safest local key setup, without putting the key in shell history:

```bash
cd /home/ayush/tau2-bench
read -rsp "OpenRouter key: " OPENROUTER_API_KEY
echo
export OPENROUTER_API_KEY
```

Alternatively, put this in the gitignored local file
`experiments/20260512-gpt55low-fullbench-harness-audit/.env.local`:

```bash
OPENROUTER_API_KEY=sk-or-...
```

The run scripts read only `OPENROUTER_API_KEY` from that file. They do not
source arbitrary shell commands from it. If `.env.local` is group/other
readable or writable, the loader prints a warning before using it.

You can create that file safely with:

```bash
cd /home/ayush/tau2-bench
experiments/20260512-gpt55low-fullbench-harness-audit/setup_openrouter_env.sh
```

The helper warns if the pasted value does not look like an OpenRouter key, but
the real authority is still `provider_preflight.py`.

If `tau2-openrouter-key-watch.service` is running, this helper restarts it after
writing `.env.local` so the preflight does not wait for the next 60-second
poll.

Or prompt for the key and immediately start the baseline:

```bash
cd /home/ayush/tau2-bench
experiments/20260512-gpt55low-fullbench-harness-audit/setup_and_run_full_baselines_openrouter.sh
```

The direct setup-and-run wrapper stops the watcher after the key prompt succeeds,
then launches the baseline itself.

Before launching the expensive run, check provider readiness:

```bash
uv run python experiments/20260512-gpt55low-fullbench-harness-audit/provider_preflight.py
```

This tests both the GPT-5.5 low chat call and the same hybrid-retriever path
the custom harness uses, including local index validation and one OpenRouter
query embedding.

After adding the OpenRouter key:

```bash
cd /home/ayush/tau2-bench
experiments/20260512-gpt55low-fullbench-harness-audit/run_full_baselines_openrouter.sh
```

If the provider may become ready later and you want a watcher instead:

```bash
cd /home/ayush/tau2-bench
INTERVAL_SECONDS=60 \
  experiments/20260512-gpt55low-fullbench-harness-audit/wait_for_provider_and_run.sh
```

The watcher only starts the benchmark after `provider_preflight.py` passes.

If the key itself may be added later, use the key-and-provider watcher:

```bash
cd /home/ayush/tau2-bench
INTERVAL_SECONDS=60 \
  experiments/20260512-gpt55low-fullbench-harness-audit/wait_for_key_and_provider_and_run.sh
```

That watcher waits for a key already present in its own process environment or
for `.env.local`, then waits for provider preflight, then starts the same full
baseline script. If the watcher is already running as systemd, use
`setup_openrouter_env.sh`; a new `export` in another shell will not update an
already-running service process.

To run that watcher as a user-systemd service:

```bash
cd /home/ayush/tau2-bench
experiments/20260512-gpt55low-fullbench-harness-audit/start_key_provider_watcher.sh
```

The service name is `tau2-openrouter-key-watch.service`; output appends to
`key_provider_watcher.stdout.log`. The systemd helper defaults to a 60-second
poll interval unless `INTERVAL_SECONDS` is set.

The `start_key_provider_watcher.sh` helper creates a transient service for the
current user-systemd session. To install the same watcher as a normal enabled
user service that starts again on later user sessions:

```bash
cd /home/ayush/tau2-bench
experiments/20260512-gpt55low-fullbench-harness-audit/install_key_provider_watcher_service.sh
```

The script:

- preflights one GPT-5.5 low call with `max_tokens=768`
- preflights the custom harness hybrid-retriever path
- runs the default Tau harness with `--auto-resume`
- runs the four custom batch-subagent passes with `--auto-resume`
- rebuilds `comparison_summary.json` and `comparison_summary.csv`

It does not store API keys. `.env.local` is ignored and should stay local.

## Completion Gate

Current status:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/status.py
```

After the run:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/check_completeness.py
```

This must print `complete` before failure analysis or prompt/tool-description fixes start.
It checks more than counts: exact source labels, model, reasoning effort,
`max_tokens`, temperature, max steps/errors, default retrieval mode, custom
hybrid retrieval, seeds, and harness identities. This is meant to catch stale
same-prefix artifacts before we trust the audit.

## Failure Packets

After `comparison_summary.json` exists and the completion gate passes:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/generate_failure_packets.py
```

This writes per-task review packets under `failure_packets/` for one-by-one
human/Codex audit. The packets include expected actions, reward basis, custom
failed runs, observed custom tool calls, compact custom/default message
timelines, default pass counts, and evidence paths. By default it writes one
packet for every task, not only failed tasks, because the final audit table must
cover all 97 tasks.

If `check_completeness.py` fails, the generator refuses to write final packets.
For a dry run while the benchmark is still incomplete:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/generate_failure_packets.py --allow-incomplete
```

Those packets go under `failure_packets_incomplete/` and are marked
`INCOMPLETE`.

After every packet has been manually/Codex reviewed and all five audit-note
fields are filled in:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/compile_failure_audit.py
```

This compiles:

- `failure_audit.md`
- `failure_audit.json`
- `generic_fix_input.md`
- `generic_fix_input.json`

The compiler refuses incomplete packets, packets missing from the manifest,
extra Markdown packets not listed in the manifest, and blank or placeholder
audit-note fields. It only compiles reviewer-written notes and keeps the fix
surface generic: no task-specific prompt fixes, no hidden-answer leaks, and no
harness behavior changes.

Use `failure_audit.*` for diagnosis. Use `generic_fix_input.*` as the
audit-derived handoff while editing prompts/tool descriptions. It intentionally
omits task ids, expected actions, reward labels, evidence paths, message
timelines, and customer data.

## Fix-Branch Guard

After creating the prompt/tool-description fix branch:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/check_prompt_only_diff.py --base main
```

This is only a guardrail. It checks committed branch changes, staged changes,
and unstaged working-tree changes, then verifies that changed paths stay within
the prompt and tool-description surfaces and flags code-looking diff lines for
manual review. It also requires the sanitized `generic_fix_input.json` handoff
and uses the raw `failure_audit.json` only to reject exact task-specific leaks
in the diff. It does not replace the anti-cheating review.

## Post-Fix Custom Rerun

After the failure audit is complete, the prompt/tool-description branch is
created, and `check_prompt_only_diff.py` passes:

```bash
experiments/20260512-gpt55low-fullbench-harness-audit/run_postfix_custom_openrouter.sh
```

This runs the custom batch-subagent harness four times per task with the same
seeds and writes `postfix_comparison_summary.json/csv`.

## Final Completion Audit

Before calling the objective done:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/completion_audit.py
```

This checks the full chain: baseline 4x custom/default run, per-task audit,
non-main prompt/tool-description fix branch, post-fix 4x custom rerun, and
post-fix accuracy measurement.

## Offline Scaffold Self-Test

To run the whole local diagnostic bundle:

```bash
experiments/20260512-gpt55low-fullbench-harness-audit/doctor.sh
```

This runs syntax checks, `selftest.py`, current status, completion audit, and
provider preflight if `OPENROUTER_API_KEY` is set. It does not start the
benchmark.

After editing the audit helpers:

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/selftest.py
```

This checks provider-error redaction, env-loader safety, explicit prefix
override behavior, and source-prefix rejection in the completeness gate. It
does not call any model.

## Analysis Outputs

```bash
python3 experiments/20260512-gpt55low-fullbench-harness-audit/analyze_comparison.py
```

Writes:

- `comparison_summary.json`
- `comparison_summary.csv`

By default, it only includes run names for this experiment:

- custom prefix: `baseline_custom_openrouter_gpt55low_`
- default prefix: `baseline_default_tau_bm25_openrouter_gpt55low_`
