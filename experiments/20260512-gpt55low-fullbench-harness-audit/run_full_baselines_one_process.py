#!/usr/bin/env python3
"""Run default Tau and custom banking baselines inside one Python process."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_HARNESS_DIR = REPO_ROOT / "default_harness"
CUSTOM_HARNESS_DIR = REPO_ROOT / "custom_harness"
CUSTOM_RUNNER_PATH = (
    REPO_ROOT / "experiments/20260509-gpt54mini-harness/run_azure_batch.py"
)

MODEL = "azure/gpt-5.5"
REASONING_EFFORT = "low"
MAX_TOKENS = 768
TEMPERATURE = 1.0
MAX_STEPS = 100
MAX_ERRORS = 10
TIMEOUT_SECONDS = None
CUSTOM_PARALLELISM = 388
DEFAULT_CONCURRENCY = 388
CUSTOM_PREFIX = "notime_custom_azure_gpt55low_"
DEFAULT_SAVE_TO = "notime_default_tau_bm25_azure_gpt55low_4trials_seed4101"
SUMMARY_JSON = EXPERIMENT_DIR / "notime_comparison_summary.json"
SUMMARY_CSV = EXPERIMENT_DIR / "notime_comparison_summary.csv"
CUSTOM_REPEATS = (
    (1, 849558),
    (2, 551167),
    (3, 811445),
    (4, 613921),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-file",
        type=Path,
        default=EXPERIMENT_DIR / ".env.local",
        help="Local env file containing Azure and OpenRouter keys.",
    )
    parser.add_argument("--custom-parallelism", type=int, default=CUSTOM_PARALLELISM)
    parser.add_argument("--default-concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--skip-provider-preflight", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--custom-only", action="store_true")
    parser.add_argument("--default-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task ids for a small smoke run. Defaults to all banking tasks.",
    )
    return parser.parse_args()


def import_from_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_env(env_file: Path, custom_runner: Any) -> None:
    env = os.environ.copy()
    env.update(custom_runner.load_env_file(env_file))
    env = custom_runner.patch_azure_env(env)
    custom_runner.require_model_env(MODEL, env)
    os.environ.update(env)
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("Missing OPENROUTER_API_KEY for Qwen query embeddings")


def run_provider_preflight() -> None:
    from provider_preflight import (
        preflight_chat,
        preflight_hybrid_retrieval,
        require_azure_chat_env,
        require_openrouter_embedding_key,
    )

    args = argparse.Namespace(
        chat_model=MODEL,
        embedding_model="qwen/qwen3-embedding-8b",
        max_tokens=MAX_TOKENS,
        reasoning_effort=REASONING_EFFORT,
        skip_embeddings=False,
    )
    print("[one-process] provider preflight start", flush=True)
    require_azure_chat_env()
    require_openrouter_embedding_key()
    preflight_chat(args)
    preflight_hybrid_retrieval(args)
    print("provider_preflight_ok", flush=True)


def record_launch_state() -> None:
    from record_launch_state import launch_state, repo_relative

    output = EXPERIMENT_DIR / "baseline_one_process_launch_state.json"
    payload = launch_state("baseline-one-process", output)
    output.write_text(
        __import__("json").dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(f"launch_state={repo_relative(output)}", flush=True)
    print(f"branch={payload['branch']}", flush=True)
    print(f"head_commit={payload['head_commit']}", flush=True)
    print(f"dirty={payload['dirty']}", flush=True)


def custom_args(args: argparse.Namespace) -> argparse.Namespace:
    repeats = [
        f"{CUSTOM_PREFIX}r{run_index}_s{seed}:{seed}"
        for run_index, seed in CUSTOM_REPEATS
    ]
    namespace = argparse.Namespace(
        tasks=args.tasks,
        model=MODEL,
        max_steps=MAX_STEPS,
        max_errors=MAX_ERRORS,
        max_tokens=MAX_TOKENS,
        timeout_seconds=TIMEOUT_SECONDS,
        temperature=TEMPERATURE,
        reasoning_effort=REASONING_EFFORT,
        reasoning_enabled=False,
        seed=42,
        subagent_delegation="batch",
        batch_name=None,
        repeat=repeats,
        env_file=args.env_file,
        auto_resume=True,
        parallelism=args.custom_parallelism,
        s3_uri=os.environ.get("TAU3_BENCH_S3_URI"),
        s3_strict=False,
        allow_bm25_only=False,
        provider_error_retries=4,
    )
    return namespace


def run_custom_baseline(args: argparse.Namespace, custom_runner: Any) -> int:
    print(
        "[one-process] custom start "
        f"parallelism={args.custom_parallelism} repeats={len(CUSTOM_REPEATS)}",
        flush=True,
    )
    status = custom_runner.run_batch(custom_args(args))
    print(f"[one-process] custom exit status={status}", flush=True)
    return status


def run_default_baseline(args: argparse.Namespace) -> int:
    from tau2.data_model.simulation import TextRunConfig
    from tau2.run import run_domain
    from tau2.utils.llm_utils import set_llm_log_mode

    print(
        "[one-process] default start "
        f"concurrency={args.default_concurrency} save_to={DEFAULT_SAVE_TO}",
        flush=True,
    )
    set_llm_log_mode("latest")
    llm_args = {
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_effort": REASONING_EFFORT,
    }
    config = TextRunConfig(
        domain="banking_knowledge",
        retrieval_config="bm25",
        task_ids=args.tasks,
        agent="llm_agent",
        llm_agent=MODEL,
        llm_args_agent=llm_args,
        user="user_simulator",
        llm_user=MODEL,
        llm_args_user=llm_args,
        num_trials=4,
        max_concurrency=args.default_concurrency,
        max_steps=MAX_STEPS,
        max_errors=MAX_ERRORS,
        timeout=TIMEOUT_SECONDS,
        seed=4101,
        save_to=DEFAULT_SAVE_TO,
        verbose_logs=True,
        auto_resume=True,
        log_level="ERROR",
        hallucination_retries=0,
    )
    run_domain(config)
    print("[one-process] default exit status=0", flush=True)
    return 0


def run_analysis() -> int:
    from analyze_comparison import main as analyze_main
    from check_completeness import main as completeness_main

    print("[one-process] analysis start", flush=True)
    old_argv = sys.argv[:]
    try:
        default_prefix = DEFAULT_SAVE_TO.removesuffix("4trials_seed4101")
        sys.argv = [
            "analyze_comparison.py",
            "--custom-source-prefix",
            CUSTOM_PREFIX,
            "--default-source-prefix",
            default_prefix,
            "--output-json",
            str(SUMMARY_JSON),
            "--output-csv",
            str(SUMMARY_CSV),
        ]
        analyze_status = analyze_main()
        sys.argv = [
            "check_completeness.py",
            "--summary",
            str(SUMMARY_JSON),
            "--required-custom-prefix",
            CUSTOM_PREFIX,
            "--required-default-prefix",
            default_prefix,
            "--required-default-source-label",
            DEFAULT_SAVE_TO,
        ]
        completeness_status = completeness_main()
    finally:
        sys.argv = old_argv
    if analyze_status not in (0, None):
        return int(analyze_status)
    if completeness_status not in (0, None):
        return int(completeness_status)
    print("[one-process] analysis done", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.custom_only and args.default_only:
        raise SystemExit("--custom-only and --default-only cannot both be set")

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(EXPERIMENT_DIR))
    sys.path.insert(0, str(DEFAULT_HARNESS_DIR / "src"))
    sys.path.insert(0, str(CUSTOM_HARNESS_DIR))
    sys.path.insert(0, str(REPO_ROOT))

    custom_runner = import_from_path("tau3_custom_batch_runner", CUSTOM_RUNNER_PATH)
    load_env(args.env_file, custom_runner)
    record_launch_state()
    if not args.skip_provider_preflight:
        run_provider_preflight()
    if args.dry_run:
        selected_tasks = "all" if args.tasks is None else ",".join(args.tasks)
        print(
            "[one-process] dry-run "
            f"custom_parallelism={args.custom_parallelism} "
            f"default_concurrency={args.default_concurrency} "
            f"timeout_seconds={TIMEOUT_SECONDS} "
            f"tasks={selected_tasks}",
            flush=True,
        )
        return 0

    jobs = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        if not args.default_only:
            jobs.append(("custom", executor.submit(run_custom_baseline, args, custom_runner)))
        if not args.custom_only:
            jobs.append(("default", executor.submit(run_default_baseline, args)))

        status = 0
        for future in as_completed([future for _, future in jobs]):
            name = next(label for label, item in jobs if item is future)
            try:
                job_status = future.result()
            except Exception:
                print(f"[one-process] {name} failed", file=sys.stderr, flush=True)
                traceback.print_exc()
                job_status = 1
            print(f"[one-process] job-exit {name} status={job_status}", flush=True)
            if job_status:
                status = 1

    if not args.skip_analysis and not (args.custom_only or args.default_only):
        analysis_status = run_analysis()
        if analysis_status:
            status = 1

    if status:
        print("[one-process] failed; artifacts were preserved", file=sys.stderr)
        return status
    print("[one-process] all done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
