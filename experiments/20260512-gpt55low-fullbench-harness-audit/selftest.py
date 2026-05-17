#!/usr/bin/env python3
"""Offline self-tests for the benchmark audit scaffolding."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=check,
    )


def test_provider_redaction() -> None:
    provider_preflight = load_module(
        "provider_preflight", EXPERIMENT_DIR / "provider_preflight.py"
    )
    raw = '{"user_id":"user_secret","key":"sk-or-v1-secret"}'
    redacted = provider_preflight.sanitize_error(raw)
    assert "user_secret" not in redacted
    assert "sk-or-v1-secret" not in redacted
    assert "[REDACTED]" in redacted


def test_provider_preflight_rejects_bm25_only_degrade() -> None:
    provider_preflight = load_module(
        "provider_preflight_degrade", EXPERIMENT_DIR / "provider_preflight.py"
    )

    class FakeEmbeddings:
        shape = (1, 3)

    class FakeRetriever:
        def __init__(self, *, embedding_model: str) -> None:
            self.embedding_model = embedding_model
            self.embeddings = FakeEmbeddings()

        def _query_embedding(self, query: str):
            return None

        def search(self, query: str, top_k: int):
            raise AssertionError("preflight accepted BM25-only fallback")

    args = type("Args", (), {"embedding_model": "fake-embedding-model"})()
    package_module = types.ModuleType("tau3_custom_harness")
    retrieval_module = types.ModuleType("tau3_custom_harness.retrieval")
    retrieval_module.BankingHybridRetriever = FakeRetriever
    original_package = sys.modules.get("tau3_custom_harness")
    original_retrieval = sys.modules.get("tau3_custom_harness.retrieval")
    sys.modules["tau3_custom_harness"] = package_module
    sys.modules["tau3_custom_harness.retrieval"] = retrieval_module
    try:
        try:
            provider_preflight.preflight_hybrid_retrieval(args)
        except RuntimeError as exc:
            assert "could not fetch a query embedding" in str(exc)
        else:
            raise AssertionError("preflight accepted missing query embeddings")
    finally:
        if original_package is None:
            sys.modules.pop("tau3_custom_harness", None)
        else:
            sys.modules["tau3_custom_harness"] = original_package
        if original_retrieval is None:
            sys.modules.pop("tau3_custom_harness.retrieval", None)
        else:
            sys.modules["tau3_custom_harness.retrieval"] = original_retrieval


def test_provider_preflight_missing_key_is_clean() -> None:
    env = dict(os.environ)
    env["AZURE_OPENAI_API_KEY"] = "test-azure-key"
    env["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com"
    env["AZURE_OPENAI_API_VERSION"] = "2025-04-01-preview"
    env.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        ["python3", str(EXPERIMENT_DIR / "provider_preflight.py")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "preflight_failed RuntimeError: OPENROUTER_API_KEY is not set" in combined
    assert "Traceback" not in combined


def test_provider_preflight_rejects_wrong_key_shape_is_clean() -> None:
    env = dict(os.environ)
    env["AZURE_OPENAI_API_KEY"] = "test-azure-key"
    env["AZURE_OPENAI_ENDPOINT"] = "https://example.openai.azure.com"
    env["AZURE_OPENAI_API_VERSION"] = "2025-04-01-preview"
    env["OPENROUTER_API_KEY"] = "not-an-openrouter-key"
    result = subprocess.run(
        ["python3", str(EXPERIMENT_DIR / "provider_preflight.py")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )
    combined = result.stdout + result.stderr
    assert result.returncode != 0
    assert "expected a value starting with sk-or-" in combined
    assert "not-an-openrouter-key" not in combined
    assert "Traceback" not in combined


def test_openrouter_env_loader_reads_only_key() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        env_path = tmp_path / ".env.local"
        side_effect_path = tmp_path / "side-effect"
        env_path.write_text(
            "\n".join(
                [
                    "IGNORED_VALUE=1",
                    f"touch {side_effect_path}",
                    'OPENROUTER_API_KEY="test-openrouter-key-loader"',
                ]
            ),
            encoding="utf-8",
        )
        env_path.chmod(0o644)
        result = subprocess.run(
            [
                "bash",
                "-c",
                "\n".join(
                    [
                        "unset OPENROUTER_API_KEY",
                        f"source {EXPERIMENT_DIR / 'load_openrouter_env.sh'}",
                        f"OPENROUTER_ENV_FILE={env_path}",
                        f"load_openrouter_env {EXPERIMENT_DIR}",
                        'printf "%s\\n" "${OPENROUTER_API_KEY:-missing}"',
                    ]
                ),
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        assert result.stdout.strip() == "test-openrouter-key-loader"
        assert "chmod 600" in result.stderr
        assert not side_effect_path.exists()


def test_setup_openrouter_env_warns_on_wrong_key_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / ".env.local"
        env = dict(os.environ)
        env["OPENROUTER_ENV_FILE"] = str(env_path)
        env["SETUP_OPENROUTER_NO_WATCHER_WAKE"] = "1"
        result = subprocess.run(
            [str(EXPERIMENT_DIR / "setup_openrouter_env.sh")],
            cwd=REPO_ROOT,
            input="not-an-openrouter-key\n",
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        assert env_path.read_text(encoding="utf-8") == (
            "OPENROUTER_API_KEY=not-an-openrouter-key\n"
        )
        mode = env_path.stat().st_mode & 0o777
        assert mode == 0o600
        assert "does not look like an OpenRouter key" in result.stderr


def test_setup_openrouter_env_accepts_openrouter_key_shape() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / ".env.local"
        env = dict(os.environ)
        env["OPENROUTER_ENV_FILE"] = str(env_path)
        env["SETUP_OPENROUTER_NO_WATCHER_WAKE"] = "1"
        result = subprocess.run(
            [str(EXPERIMENT_DIR / "setup_openrouter_env.sh")],
            cwd=REPO_ROOT,
            input="sk-or-v1-test-openrouter-key\n",
            text=True,
            capture_output=True,
            env=env,
            check=True,
        )
        assert env_path.read_text(encoding="utf-8") == (
            "OPENROUTER_API_KEY=sk-or-v1-test-openrouter-key\n"
        )
        assert "does not look like an OpenRouter key" not in result.stderr


def test_setup_openrouter_env_empty_input_preserves_existing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        env_path = Path(tmp) / ".env.local"
        original = "OPENROUTER_API_KEY=sk-or-v1-existing-test-key\n"
        env_path.write_text(original, encoding="utf-8")
        env_path.chmod(0o600)
        env = dict(os.environ)
        env["OPENROUTER_ENV_FILE"] = str(env_path)
        env["SETUP_OPENROUTER_NO_WATCHER_WAKE"] = "1"
        result = subprocess.run(
            [str(EXPERIMENT_DIR / "setup_openrouter_env.sh")],
            cwd=REPO_ROOT,
            input="\n",
            text=True,
            capture_output=True,
            env=env,
            check=False,
        )
        assert result.returncode == 2
        assert env_path.read_text(encoding="utf-8") == original
        assert "No key entered" in result.stderr


def test_status_reports_watcher_durability() -> None:
    status = load_module("status_watcher_labels", EXPERIMENT_DIR / "status.py")
    original_run_text = status.run_text
    original_process_running = status.key_provider_watcher_process_running
    original_user = os.environ.get("USER")

    def fake_run_text(command: list[str]) -> str:
        if command[:3] == ["systemctl", "--user", "show"]:
            return "\n".join(
                [
                    "ActiveState=active",
                    "FragmentPath=/home/ayush/.config/systemd/user/tau2-openrouter-key-watch.service",
                ]
            )
        if command == ["systemctl", "--user", "is-enabled", status.WATCHER_UNIT]:
            return "enabled"
        if command[:3] == ["loginctl", "show-user", "ayush"]:
            return "Linger=yes"
        return ""

    try:
        status.run_text = fake_run_text
        status.key_provider_watcher_process_running = lambda: True
        os.environ["USER"] = "ayush"
        running, label = status.key_provider_watcher_status()
        assert running
        assert label == "running (enabled user service; linger enabled)"

        def transient_run_text(command: list[str]) -> str:
            if command[:3] == ["systemctl", "--user", "show"]:
                return "\n".join(
                    [
                        "ActiveState=active",
                        "FragmentPath=/run/user/1000/systemd/transient/tau2-openrouter-key-watch.service",
                    ]
                )
            if command == ["systemctl", "--user", "is-enabled", status.WATCHER_UNIT]:
                return "transient"
            return ""

        status.run_text = transient_run_text
        running, label = status.key_provider_watcher_status()
        assert running
        assert label == "running (transient service)"
    finally:
        status.run_text = original_run_text
        status.key_provider_watcher_process_running = original_process_running
        if original_user is None:
            os.environ.pop("USER", None)
        else:
            os.environ["USER"] = original_user


def test_status_reports_latest_provider_failure_safely() -> None:
    status = load_module("status_provider_failure", EXPERIMENT_DIR / "status.py")
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "watcher.log"
        fake_user = "user-secret-123"
        fake_openrouter_key = "sk-or-v1-secret-test-key"
        fake_openai_key = "sk-proj-secret-test-key"
        log_path.write_text(
            "\n".join(
                [
                    "preflight_failed RuntimeError: older failure",
                    (
                        "preflight_failed AuthenticationError: "
                        f'{{"user_id":"{fake_user}","key":"{fake_openrouter_key}",'
                        f'"fallback":"{fake_openai_key}"}}'
                    ),
                ]
            ),
            encoding="utf-8",
        )

        failure = status.last_provider_preflight_failure(log_path)
        assert "preflight_failed AuthenticationError" in failure
        assert "older failure" not in failure
        assert fake_user not in failure
        assert fake_openrouter_key not in failure
        assert fake_openai_key not in failure
        assert "[REDACTED]" in failure
        assert "[REDACTED_OPENROUTER_KEY]" in failure
        assert "[REDACTED_OPENAI_KEY]" in failure

        assert status.last_provider_preflight_failure(Path(tmp) / "missing.log") == ""


def test_analyzer_prefix_override() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_json = Path(tmp) / "summary.json"
        output_csv = Path(tmp) / "summary.csv"
        run(
            [
                "python3",
                str(EXPERIMENT_DIR / "analyze_comparison.py"),
                "--custom-source-prefix",
                "postfix_custom_openrouter_gpt55low_",
                "--default-source-prefix",
                "baseline_default_tau_bm25_openrouter_gpt55low_",
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
            ]
        )
        payload = json.loads(output_json.read_text(encoding="utf-8"))
        assert payload["inputs"]["custom_source_prefixes"] == [
            "postfix_custom_openrouter_gpt55low_"
        ]
        assert payload["inputs"]["default_source_prefixes"] == [
            "baseline_default_tau_bm25_openrouter_gpt55low_"
        ]


def test_completeness_rejects_wrong_prefix() -> None:
    result = run(
        [
            "python3",
            str(EXPERIMENT_DIR / "check_completeness.py"),
            "--required-custom-prefix",
            "wrong_prefix_",
        ],
        check=False,
    )
    assert result.returncode != 0
    assert "invalid custom source prefixes" in result.stdout


def test_completeness_validates_contract() -> None:
    check_completeness = load_module(
        "check_completeness", EXPERIMENT_DIR / "check_completeness.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "summary.json"
        rows = []
        custom_labels = sorted(
            check_completeness.custom_source_labels_for_prefix(
                check_completeness.DEFAULT_REQUIRED_CUSTOM_PREFIX
            )
        )
        for task_id in check_completeness.expected_task_ids():
            rows.append(
                {
                    "task_id": task_id,
                    "custom_pass_count": 0,
                    "default_pass_count": 0,
                    "custom_runs": [
                        fake_run(label, check_completeness.EXPECTED_CUSTOM_CONTRACT)
                        for label in custom_labels
                    ],
                    "default_runs": [
                        fake_run(
                            check_completeness.EXPECTED_DEFAULT_SOURCE_LABEL,
                            check_completeness.EXPECTED_DEFAULT_CONTRACT,
                            run_id=(
                                f"{check_completeness.EXPECTED_DEFAULT_SOURCE_LABEL}"
                                f"_trial{index + 1}"
                            ),
                        )
                        for index in range(check_completeness.EXPECTED_RUNS_PER_HARNESS)
                    ],
                }
            )
        payload = {
            "inputs": {
                "custom_source_prefixes": [
                    check_completeness.DEFAULT_REQUIRED_CUSTOM_PREFIX
                ],
                "default_source_prefixes": [
                    check_completeness.DEFAULT_REQUIRED_DEFAULT_PREFIX
                ],
            },
            "tasks": rows,
        }
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        good = run(
            [
                "python3",
                str(EXPERIMENT_DIR / "check_completeness.py"),
                "--summary",
                str(summary_path),
            ],
            check=False,
        )
        assert good.returncode == 0, good.stdout + good.stderr

        duplicate_payload = json.loads(json.dumps(payload))
        duplicate_payload["tasks"][0]["custom_runs"][1]["run_id"] = (
            duplicate_payload["tasks"][0]["custom_runs"][0]["run_id"]
        )
        summary_path.write_text(json.dumps(duplicate_payload), encoding="utf-8")
        duplicate = run(
            [
                "python3",
                str(EXPERIMENT_DIR / "check_completeness.py"),
                "--summary",
                str(summary_path),
            ],
            check=False,
        )
        assert duplicate.returncode != 0
        assert "duplicate custom run ids" in duplicate.stdout

        payload["tasks"][0]["custom_runs"][0]["run_contract"]["max_tokens"] = 512
        summary_path.write_text(json.dumps(payload), encoding="utf-8")
        bad = run(
            [
                "python3",
                str(EXPERIMENT_DIR / "check_completeness.py"),
                "--summary",
                str(summary_path),
            ],
            check=False,
        )
        assert bad.returncode != 0
        assert "invalid custom contract" in bad.stdout


def test_completeness_requires_exact_task_ids() -> None:
    check_completeness = load_module(
        "check_completeness_task_ids", EXPERIMENT_DIR / "check_completeness.py"
    )
    expected_ids = check_completeness.expected_task_ids()
    rows = [{"task_id": task_id} for task_id in expected_ids]
    errors: list[str] = []
    check_completeness.check_task_coverage(errors, rows)
    assert errors == []

    rows[-1]["task_id"] = rows[0]["task_id"]
    errors = []
    check_completeness.check_task_coverage(errors, rows)
    assert any("duplicate task rows" in error for error in errors)
    assert any("missing task rows" in error for error in errors)


def test_prompt_guard_detects_task_data() -> None:
    guard = load_module(
        "check_prompt_only_diff", EXPERIMENT_DIR / "check_prompt_only_diff.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        audit_path = Path(tmp) / "failure_audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "tasks": [
                        {
                            "task_id": "task_042",
                            "reward_basis": ["reward-secret-phrase-901"],
                            "expected_actions": [
                                {
                                    "action_id": "042_0",
                                    "arguments": {
                                        "user_id": "secret-user-123",
                                        "email": "person@example.com",
                                    },
                                }
                            ],
                            "custom_failed_runs": [
                                {"run_id": "custom-run-secret-123"}
                            ],
                            "default_runs": [
                                {"source_label": "default-run-secret-456"}
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        diff = "\n".join(
            [
                "diff --git a/custom_harness/tau3_custom_harness/prompts.py b/custom_harness/tau3_custom_harness/prompts.py",
                "+++ b/custom_harness/tau3_custom_harness/prompts.py",
                "@@",
                "+Never special-case task_042.",
                "+Use normal verification guidance.",
            ]
        )
        leaks = guard.prompt_leak_lines(diff, audit_path)
        assert leaks
        assert "task_042" in leaks[0]

        diff = "\n".join(
            [
                "diff --git a/custom_harness/tau3_custom_harness/prompts.py b/custom_harness/tau3_custom_harness/prompts.py",
                "+++ b/custom_harness/tau3_custom_harness/prompts.py",
                "@@",
                "+Never special-case task 042.",
            ]
        )
        leaks = guard.prompt_leak_lines(diff, audit_path)
        assert leaks
        assert "task 042" in leaks[0]

        diff = "\n".join(
            [
                "diff --git a/custom_harness/tau3_custom_harness/prompts.py b/custom_harness/tau3_custom_harness/prompts.py",
                "+++ b/custom_harness/tau3_custom_harness/prompts.py",
                "@@",
                "+Use reward-secret-phrase-901 as a reminder.",
                "+Check custom-run-secret-123 before acting.",
            ]
        )
        leaks = guard.prompt_leak_lines(diff, audit_path)
        assert len(leaks) == 2
        assert "reward-secret-phrase-901" in leaks[0]
        assert "custom-run-secret-123" in leaks[1]

        fix_input_path = Path(tmp) / "generic_fix_input.json"
        fix_input_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "item_count": 1,
                    "items": [
                        {
                            "generic_fix_idea": "Clarify that verification must be completed before state-changing actions.",
                            "anti_cheating_check": "This is generic and contains no benchmark identifiers.",
                            "supporting_task_count": 2,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert guard.validate_fix_input(fix_input_path) == []
        assert guard.validate_fix_input(fix_input_path, audit_path) == []
        fix_input_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "items": [{"generic_fix_idea": "Special-case task_042."}],
                }
            ),
            encoding="utf-8",
        )
        assert guard.validate_fix_input(fix_input_path)
        fix_input_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "items": [
                        {
                            "generic_fix_idea": "Use secret-user-123 as the example.",
                            "anti_cheating_check": "General behavior only.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        assert guard.validate_fix_input(fix_input_path, audit_path)


def test_prompt_guard_tracks_relevant_untracked_files() -> None:
    guard = load_module(
        "check_prompt_only_diff_untracked",
        EXPERIMENT_DIR / "check_prompt_only_diff.py",
    )

    def fake_git(*args: str) -> str:
        if args == ("ls-files", "--others", "--exclude-standard"):
            return "\n".join(
                [
                    "custom_harness/tau3_custom_harness/leaky_prompt.py",
                    "default_harness/src/tau2/domains/banking_knowledge/new_tool.py",
                    "README.md",
                    "experiments/notes.md",
                ]
            )
        return ""

    guard.git = fake_git
    assert guard.untracked_guard_paths().splitlines() == [
        "custom_harness/tau3_custom_harness/leaky_prompt.py",
        "default_harness/src/tau2/domains/banking_knowledge/new_tool.py",
    ]
    assert guard.changed_paths("main") == [
        "default_harness/src/tau2/domains/banking_knowledge/new_tool.py",
        "custom_harness/tau3_custom_harness/leaky_prompt.py",
    ]


def test_generate_packets_checks_requested_summary() -> None:
    generator = load_module(
        "generate_failure_packets", EXPERIMENT_DIR / "generate_failure_packets.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        summary_path = Path(tmp) / "missing-summary.json"
        completeness = generator.run_completeness_check(summary_path)
        assert not completeness.complete
        assert str(summary_path) in completeness.output


def test_sanitized_fix_input_omits_raw_task_data() -> None:
    compiler = load_module(
        "compile_failure_audit", EXPERIMENT_DIR / "compile_failure_audit.py"
    )
    payload = {
        "generated_at": "2026-05-12T00:00:00+00:00",
        "tasks": [
            {
                "task_id": "task_042",
                "expected_actions": [{"arguments": {"user_id": "secret-user-123"}}],
                "audit_notes": {
                    "general_prompt_or_tool_description_fix_idea": "Clarify that agents must gather policy evidence before mutating state.",
                    "anti_cheating_check": "General behavior only; no task-specific IDs.",
                },
            }
        ],
    }
    sanitized = compiler.sanitized_fix_input_payload(payload)
    compiler.validate_sanitized_fix_input(sanitized)
    text = json.dumps(sanitized)
    assert "task_042" not in text
    assert "secret-user-123" not in text
    assert "expected_actions" not in text


def test_completion_audit_rejects_fix_input_task_data() -> None:
    completion_audit = load_module(
        "completion_audit_fix_input", EXPERIMENT_DIR / "completion_audit.py"
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        failure_audit = tmp_path / "failure_audit.json"
        generic_json = tmp_path / "generic_fix_input.json"
        generic_md = tmp_path / "generic_fix_input.md"
        failure_audit.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "tasks": [
                        {
                            "task_id": "task_042",
                            "expected_actions": [
                                {"arguments": {"user_id": "secret-user-123"}}
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        generic_json.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "items": [
                        {
                            "generic_fix_idea": "Use secret-user-123 as a reminder.",
                            "anti_cheating_check": "General behavior only.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        generic_md.write_text("# Generic Fix Input\n", encoding="utf-8")
        completion_audit.FAILURE_AUDIT_JSON = failure_audit
        completion_audit.GENERIC_FIX_INPUT_JSON = generic_json
        completion_audit.GENERIC_FIX_INPUT_MD = generic_md
        check = completion_audit.check_generic_fix_input()
        assert not check.passed
        assert "secret-user-123" in check.evidence


def test_launch_state_validation() -> None:
    recorder = load_module(
        "record_launch_state", EXPERIMENT_DIR / "record_launch_state.py"
    )
    completion_audit = load_module(
        "completion_audit_launch_state", EXPERIMENT_DIR / "completion_audit.py"
    )
    payload = recorder.launch_state("baseline", EXPERIMENT_DIR / "baseline_launch_state.json")
    assert payload["schema_version"] == 1
    assert payload["label"] == "baseline"
    assert payload["head_commit"]
    assert isinstance(payload["status_short"], list)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "launch_state.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert completion_audit.validate_launch_state(path, "baseline") == []
        payload["label"] = "wrong"
        path.write_text(json.dumps(payload), encoding="utf-8")
        assert completion_audit.validate_launch_state(path, "baseline")


def test_compiler_blocks_reward_basis_and_run_ids_in_fix_notes() -> None:
    compiler = load_module(
        "compile_failure_audit_blocked_terms",
        EXPERIMENT_DIR / "compile_failure_audit.py",
    )
    packet = compiler.Packet(
        path=Path("/tmp/task_042.md"),
        task_id="task_042",
        packet_status="complete",
        counts={},
        expected_actions=[],
        reward_basis=["reward-secret-phrase-901"],
        custom_failed_runs=[{"run_id": "custom-run-secret-123"}],
        default_runs=[],
        evidence_paths=[],
        audit_notes={
            "general_prompt_or_tool_description_fix_idea": (
                "Add reward-secret-phrase-901 to the prompt."
            ),
            "anti_cheating_check": "General behavior only.",
        },
    )
    try:
        compiler.validate_fix_notes_are_general(packet)
    except SystemExit as exc:
        assert "reward-secret-phrase-901" in str(exc)
    else:
        raise AssertionError("compiler accepted reward-basis text in fix notes")

    packet = compiler.Packet(
        path=Path("/tmp/task_043.md"),
        task_id="task_043",
        packet_status="complete",
        counts={},
        expected_actions=[],
        reward_basis=[],
        custom_failed_runs=[{"run_id": "custom-run-secret-123"}],
        default_runs=[],
        evidence_paths=[],
        audit_notes={
            "general_prompt_or_tool_description_fix_idea": (
                "Add custom-run-secret-123 to the prompt."
            ),
            "anti_cheating_check": "General behavior only.",
        },
    )
    try:
        compiler.validate_fix_notes_are_general(packet)
    except SystemExit as exc:
        assert "custom-run-secret-123" in str(exc)
    else:
        raise AssertionError("compiler accepted run id text in fix notes")


def test_compiler_requires_codex_review_confirmation() -> None:
    compiler = load_module(
        "compile_failure_audit_codex_confirmation",
        EXPERIMENT_DIR / "compile_failure_audit.py",
    )
    packet = compiler.Packet(
        path=Path("/tmp/task_042.md"),
        task_id="task_042",
        packet_status="complete",
        counts={},
        expected_actions=[],
        reward_basis=[],
        custom_failed_runs=[],
        default_runs=[],
        evidence_paths=[],
        audit_notes={"codex_review_confirmation": "A reviewer looked at it."},
    )
    try:
        compiler.validate_codex_review_confirmation(packet)
    except SystemExit as exc:
        assert "Codex review confirmation" in str(exc) or "weak" in str(exc)
    else:
        raise AssertionError("compiler accepted weak Codex review confirmation")

    packet.audit_notes["codex_review_confirmation"] = (
        "Codex personally reviewed this task one by one."
    )
    compiler.validate_codex_review_confirmation(packet)


def test_packet_coverage_requires_97_unique_tasks() -> None:
    compiler = load_module(
        "compile_failure_audit_coverage", EXPERIMENT_DIR / "compile_failure_audit.py"
    )
    packets = [
        compiler.Packet(
            path=Path(f"/tmp/task_{index:03d}.md"),
            task_id=f"task_{index:03d}",
            packet_status="complete",
            counts={},
            expected_actions=[],
            reward_basis=[],
            custom_failed_runs=[],
            default_runs=[],
            evidence_paths=[],
            audit_notes={},
        )
        for index in range(1, 98)
    ]
    compiler.validate_packet_coverage(packets)
    try:
        compiler.validate_packet_coverage(packets[:-1])
    except SystemExit:
        pass
    else:
        raise AssertionError("coverage check accepted missing packet")


def fake_run(source_label: str, contract: dict, *, run_id: str | None = None) -> dict:
    return {
        "run_id": run_id or source_label,
        "source_label": source_label,
        "reward": 0.0,
        "passed": False,
        "termination_reason": "user_stop",
        "error_type": None,
        "run_contract": json.loads(json.dumps(contract)),
    }


def main() -> int:
    test_provider_redaction()
    test_provider_preflight_rejects_bm25_only_degrade()
    test_provider_preflight_missing_key_is_clean()
    test_provider_preflight_rejects_wrong_key_shape_is_clean()
    test_openrouter_env_loader_reads_only_key()
    test_setup_openrouter_env_warns_on_wrong_key_shape()
    test_setup_openrouter_env_accepts_openrouter_key_shape()
    test_setup_openrouter_env_empty_input_preserves_existing_file()
    test_status_reports_watcher_durability()
    test_status_reports_latest_provider_failure_safely()
    test_analyzer_prefix_override()
    test_completeness_rejects_wrong_prefix()
    test_completeness_validates_contract()
    test_completeness_requires_exact_task_ids()
    test_prompt_guard_detects_task_data()
    test_prompt_guard_tracks_relevant_untracked_files()
    test_generate_packets_checks_requested_summary()
    test_sanitized_fix_input_omits_raw_task_data()
    test_completion_audit_rejects_fix_input_task_data()
    test_launch_state_validation()
    test_compiler_blocks_reward_basis_and_run_ids_in_fix_notes()
    test_compiler_requires_codex_review_confirmation()
    test_packet_coverage_requires_97_unique_tasks()
    print("selftest_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
