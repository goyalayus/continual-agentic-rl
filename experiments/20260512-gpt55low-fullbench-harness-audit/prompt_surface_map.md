# Prompt And Tool-Description Surface Map

This note maps the safe edit surface for the later fix branch. It is not an
audit of benchmark failures, and it does not contain task-specific fixes,
expected actions, customer data, hidden arguments from traces, or raw failure
packet content.

The future fix branch must still be reviewed by `check_prompt_only_diff.py`.
This map is just a reading guide so prompt/tool-description edits stay
general.

## Allowed Paths

`check_prompt_only_diff.py` currently allows changes only in:

- `tau3_custom_harness/prompts.py`
- `tau3_custom_harness/agent.py`
- `src/tau2/domains/banking_knowledge/tools.py`

The allowlist is path-level, but the practical safe surface is narrower:
prompt literals, tool docstrings, and correction text. Logic changes, runner
changes, retriever changes, evaluator changes, and task-specific prompt text are
out of scope for the fix branch.

## `tau3_custom_harness/prompts.py`

This is the cleanest prompt-only surface.

- `PLANNER_INSTRUCTIONS_TEMPLATE` at lines 3-99 controls the planner role,
  default-vs-discoverable tool behavior, KB delegation rules, turn shape,
  verification defaults, and the general work pattern.
- `BATCH_DELEGATION_INSTRUCTIONS` and `SINGLE_DELEGATION_INSTRUCTIONS` at
  lines 102-120 control how the planner is taught to delegate KB research.
- `BATCH_PLANNER_EXAMPLES` and `SINGLE_PLANNER_EXAMPLES` at lines 123-257 are
  synthetic few-shot examples for planner behavior.
- `SUBAGENT_INSTRUCTIONS_TEMPLATE` at lines 260-368 controls the KB subagent
  role, search/read workflow, result shape, and hidden-tool reporting style.
- `planner_instructions`, `subagent_instructions`, `planner_system_prompt`,
  and `subagent_system_prompt` at lines 371-430 assemble the final system
  prompts. Treat these as wiring. Prefer editing the templates above, not the
  assembly logic.

Use this file for broad behavioral instructions such as when to research,
when to verify, how to separate user-side and agent-side tools, and how to avoid
acting without source evidence.

## `tau3_custom_harness/agent.py`

This file contains model-facing text and tool descriptions, but it also
contains harness logic. The safe edits here are the string/docstring surfaces,
not control flow.

- `system_prompt` and `_format_default_user_tools` at lines 109-135 inject the
  planner prompt and default user-tool descriptions into the system message.
  Treat this as wiring unless a future prompt-only review finds a description
  formatting issue.
- Planner retry/correction messages at lines 177-244 are shown to the model
  when it sends an invalid assistant turn, mixes KB delegation with public
  banking tools, tries a discoverable tool without KB evidence, or tries a
  state-changing tool after a failed policy lookup.
- `ask_knowledge_subagent` and `ask_knowledge_subagents` docstrings at lines
  267-299 are planner-visible descriptions for KB delegation.
- `_run_subagent` at lines 401-425 adds the fixed research wrapper around each
  subagent request. Avoid changing this unless the final diff is purely prompt
  text.
- `search` and `read_doc` docstrings at lines 490-548 are subagent-visible
  descriptions for KB lookup.
- `knowledge_evidence_report` and the discoverable-tool evidence checks at
  lines 560-568 and 700-736 explain the evidence mechanism. These are useful
  for reading, but the fix branch should not change the mechanism.

Use this file for clearer tool descriptions and correction text only. If a diff
adds or changes Python control flow here, assume it is outside the intended
fix surface until proven otherwise.

## `src/tau2/domains/banking_knowledge/tools.py`

This is the banking tool catalog. It is allowed because many tool descriptions
and discoverable-tool cards come directly from docstrings.

- `parse_discoverable_tool_docstring` at lines 111-188 parses discoverable
  tool docstrings. The first paragraph becomes the description, `Args:` becomes
  the parameter schema, and `Returns:` becomes the success-message hint.
- `format_discoverable_tool_for_agent` at lines 191-216 formats that parsed
  docstring into the tool card the agent sees.
- `KnowledgeTools` at lines 345-356 states that agent discoverable-tool
  docstrings are the source of truth for descriptions and parameters.
- Public agent-tool docstrings at lines 376-713 describe normal banking tools
  such as transfer, user lookup, verification logging, discoverable-user-tool
  handoff, discoverable-agent-tool unlock/call, and discoverable-agent-tool
  listing.
- Agent-side discoverable-tool docstrings span roughly lines 736-3947. These
  descriptions are exposed only after the corresponding tool is discovered and
  unlocked.
- `KnowledgeUserTools` starts around line 4090. User-side discoverable-tool
  docstrings span roughly lines 4140-4282, and default user-side public tool
  docstrings span roughly lines 4359-4533.

Use this file only for general docstring clarity: what the tool does, who should
perform it, what each argument means, and what source evidence is required
before use. Do not add benchmark task IDs, expected action IDs, customer
records, trace-derived argument values, or failure-specific examples.

## Guardrails

- Run `check_prompt_only_diff.py` on the fix branch before any post-fix rerun.
- Use only the sanitized `generic_fix_input.json`/`.md` as audit-derived input
  for edits. The raw `failure_audit.*` files are for diagnosis and leak checks,
  not for copy-pasting into prompts.
- Keep changes general enough that they would still make sense on unseen
  banking tasks.
- Do not edit benchmark tasks, evaluators, runner settings, retrieval logic,
  seeds, model settings, or scoring code.
- If the diff contains code-looking changes in an allowed file, treat the guard
  failure as a serious review item, even if it might be a false positive.
