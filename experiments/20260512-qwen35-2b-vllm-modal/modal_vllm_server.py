"""Serve Qwen3.5-2B on Modal with vLLM's OpenAI-compatible API.

This file is intentionally small. It is a staging harness for smoke-testing the
model endpoint before we wire it into the Tau3 banking harness.
"""

from __future__ import annotations

import json
import asyncio
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


APP_NAME = "qwen35-2b-vllm-tau3"
MODEL_ID = "Qwen/Qwen3.5-2B"
SERVED_MODEL_NAME = "qwen3.5-2b"
VLLM_PORT = 8000
STARTUP_TIMEOUT_SECONDS = 15 * 60

GPU_TYPE = "T4"
MAX_MODEL_LEN = 8_192
GPU_MEMORY_UTILIZATION = 0.88
ENABLE_PREFIX_CACHING = True
MAX_NUM_SEQS: int | None = None
MAX_NUM_BATCHED_TOKENS: int | None = None


image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.20.2", add_python="3.12")
    .entrypoint([])
    .pip_install("more-itertools")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "HF_HOME": "/root/.cache/huggingface",
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
        }
    )
)

hf_cache = modal.Volume.from_name("qwen35-2b-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen35-2b-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=20 * 60,
    scaledown_window=60,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve() -> None:
    start_vllm(enforce_eager=True)


@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=25 * 60,
    scaledown_window=60,
    max_containers=1,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT_SECONDS)
def serve_no_eager() -> None:
    start_vllm(enforce_eager=False)


def start_vllm(*, enforce_eager: bool) -> None:
    cmd = [
        "vllm",
        "serve",
        MODEL_ID,
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "auto",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--gpu-memory-utilization",
        str(GPU_MEMORY_UTILIZATION),
        "--skip-mm-profiling",
        "--limit-mm-per-prompt",
        '{"image":0,"video":0}',
        "--generation-config",
        "vllm",
        "--load-format",
        "safetensors",
        "--safetensors-load-strategy",
        "eager",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_xml",
        "--uvicorn-log-level",
        "info",
    ]
    if MAX_NUM_SEQS is not None:
        cmd.extend(["--max-num-seqs", str(MAX_NUM_SEQS)])
    if MAX_NUM_BATCHED_TOKENS is not None:
        cmd.extend(["--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS)])
    if enforce_eager:
        cmd.append("--enforce-eager")
    if ENABLE_PREFIX_CACHING:
        cmd.append("--enable-prefix-caching")
    print("Starting vLLM:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)


def post_json(url: str, payload: dict[str, Any], *, timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_json(url: str, *, timeout: int = 30) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer EMPTY"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, *, timeout: int = 30) -> str:
    request = urllib.request.Request(
        url,
        headers={"Authorization": "Bearer EMPTY"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def wait_for_health(base_url: str, *, timeout_seconds: int) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - useful for smoke output.
            last_error = exc
        time.sleep(5)
    raise RuntimeError(f"vLLM health check failed at {base_url}: {last_error}")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]


def tau_like_payload(*, request_id: int, max_tokens: int) -> dict[str, Any]:
    policy_block = (
        "You are a banking support agent. You must verify the customer's identity "
        "before taking state-changing actions. You must use tools for account data. "
        "You must explain policy constraints clearly. You must not reveal hidden "
        "database fields to the customer. You may ask a knowledge subagent for policy "
        "evidence when documentation matters. "
    )
    repeated_policy = "\n".join(policy_block for _ in range(36))
    user_text = (
        f"Customer case {request_id}: The customer says their debit card was stolen, "
        "they are traveling tomorrow, and they want the fastest replacement path. "
        "Give a concise internal next-action plan. Do not call a tool in this load "
        "test. Include verification status, policy evidence needed, allowed action, "
        "forbidden action, and the next customer-facing sentence."
    )
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {"role": "system", "content": repeated_policy},
            {"role": "user", "content": user_text},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_user_information_by_email",
                    "description": "Look up user identity fields by email.",
                    "parameters": {
                        "type": "object",
                        "properties": {"email": {"type": "string"}},
                        "required": ["email"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_knowledge_subagent",
                    "description": (
                        "Ask a knowledge-base subagent to research policy facts. "
                        "The subagent can search and read documentation, but cannot "
                        "use database tools or talk to the customer."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"question": {"type": "string"}},
                        "required": ["question"],
                    },
                },
            },
        ],
        "tool_choice": "none",
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        raw_name, raw_value = parts
        name = raw_name.split("{", 1)[0]
        if not (name.startswith("vllm:") or name.startswith("vllm_")):
            continue
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metrics[name] = metrics.get(name, 0.0) + value
    return metrics


async def sample_metrics(base_url: str, stop: asyncio.Event) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    while not stop.is_set():
        try:
            text = await asyncio.to_thread(get_text, f"{base_url}/metrics", timeout=10)
            metrics = parse_prometheus_metrics(text)
            keep = {
                key: value
                for key, value in metrics.items()
                if any(
                    needle in key
                    for needle in (
                        "num_requests",
                        "gpu_cache",
                        "kv_cache",
                        "tokens_total",
                        "request_success",
                    )
                )
            }
            samples.append({"time": time.time(), "metrics": keep})
        except Exception as exc:  # noqa: BLE001 - metrics are best effort.
            samples.append({"time": time.time(), "error": repr(exc)})
        await asyncio.sleep(1)
    return samples


async def run_one_request(
    *,
    base_url: str,
    request_id: int,
    max_tokens: int,
    timeout: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = await asyncio.to_thread(
            post_json,
            f"{base_url}/v1/chat/completions",
            tau_like_payload(request_id=request_id, max_tokens=max_tokens),
            timeout=timeout,
        )
        elapsed = time.perf_counter() - started
        usage = response.get("usage") or {}
        return {
            "ok": True,
            "request_id": request_id,
            "seconds": elapsed,
            "usage": usage,
            "finish_reason": ((response.get("choices") or [{}])[0] or {}).get(
                "finish_reason"
            ),
        }
    except Exception as exc:  # noqa: BLE001 - record failures in artifact.
        return {
            "ok": False,
            "request_id": request_id,
            "seconds": time.perf_counter() - started,
            "error": repr(exc),
        }


async def run_load_level(
    *,
    base_url: str,
    concurrency: int,
    requests: int,
    max_tokens: int,
    timeout: int,
    request_offset: int,
) -> dict[str, Any]:
    stop_metrics = asyncio.Event()
    metrics_task = asyncio.create_task(sample_metrics(base_url, stop_metrics))
    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(index: int) -> dict[str, Any]:
        async with semaphore:
            return await run_one_request(
                base_url=base_url,
                request_id=request_offset + index,
                max_tokens=max_tokens,
                timeout=timeout,
            )

    started = time.perf_counter()
    results = await asyncio.gather(*(guarded(index) for index in range(requests)))
    elapsed = time.perf_counter() - started
    stop_metrics.set()
    metric_samples = await metrics_task

    successes = [result for result in results if result.get("ok")]
    failures = [result for result in results if not result.get("ok")]
    latencies = [float(result["seconds"]) for result in successes]
    prompt_tokens = sum((result.get("usage") or {}).get("prompt_tokens") or 0 for result in successes)
    completion_tokens = sum(
        (result.get("usage") or {}).get("completion_tokens") or 0 for result in successes
    )
    total_tokens = sum((result.get("usage") or {}).get("total_tokens") or 0 for result in successes)
    metric_peaks: dict[str, float] = {}
    for sample in metric_samples:
        for key, value in (sample.get("metrics") or {}).items():
            metric_peaks[key] = max(metric_peaks.get(key, value), value)

    return {
        "concurrency": concurrency,
        "requests": requests,
        "ok_count": len(successes),
        "failure_count": len(failures),
        "wall_seconds": elapsed,
        "requests_per_second": len(successes) / elapsed if elapsed else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "completion_tokens_per_second": completion_tokens / elapsed if elapsed else None,
        "total_tokens_per_second": total_tokens / elapsed if elapsed else None,
        "p50_request_seconds": percentile(latencies, 0.50),
        "p95_request_seconds": percentile(latencies, 0.95),
        "max_request_seconds": max(latencies) if latencies else None,
        "metric_peaks": metric_peaks,
        "failures": failures[:5],
    }


def write_load_summary(result: dict[str, Any], result_path: Path) -> None:
    rows = []
    for level in result["levels"]:
        rows.append(
            "| {concurrency} | {ok_count}/{requests} | {requests_per_second:.3f} | "
            "{completion_tokens_per_second:.1f} | {total_tokens_per_second:.1f} | "
            "{p50_request_seconds:.2f} | {p95_request_seconds:.2f} | {failure_count} |".format(
                **{
                    **level,
                    "requests_per_second": level["requests_per_second"] or 0.0,
                    "completion_tokens_per_second": level[
                        "completion_tokens_per_second"
                    ]
                    or 0.0,
                    "total_tokens_per_second": level["total_tokens_per_second"]
                    or 0.0,
                    "p50_request_seconds": level["p50_request_seconds"] or 0.0,
                    "p95_request_seconds": level["p95_request_seconds"] or 0.0,
                }
            )
        )
    best = max(
        result["levels"],
        key=lambda level: level["completion_tokens_per_second"] or 0.0,
    )
    markdown = "\n".join(
        [
            f"# Load Sweep: {result['variant']}",
            "",
            f"- started: `{result['started_at']}`",
            f"- startup to health: `{result['startup_to_health_seconds']:.1f}s`",
            f"- model: `{result['model_id']}`",
            f"- GPU: `{result['gpu_type']}`",
            f"- max model len: `{result['max_model_len']}`",
            f"- enforce eager: `{result['enforce_eager']}`",
            f"- prefix caching: `{result['enable_prefix_caching']}`",
            f"- max tokens/request: `{result['max_tokens']}`",
            "",
            "## Best",
            "",
            (
                f"Concurrency `{best['concurrency']}` reached "
                f"`{(best['completion_tokens_per_second'] or 0):.1f}` completion tok/s "
                f"and `{(best['total_tokens_per_second'] or 0):.1f}` total tok/s "
                f"with `{best['failure_count']}` failures."
            ),
            "",
            "## Results",
            "",
            "| concurrency | ok | req/s | completion tok/s | total tok/s | p50 s | p95 s | failures |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            f"Raw JSON: `{result_path.name}`",
            "",
        ]
    )
    result_path.with_suffix(".md").write_text(markdown, encoding="utf-8")


@app.local_entrypoint()
def load_sweep(
    concurrency_values: str = "1,2,4,8,16,32",
    requests_per_level: int = 32,
    max_tokens: int = 128,
    timeout: int = 180,
    variant: str = "eager_prefix_auto_kv",
    no_eager: bool = False,
) -> None:
    """Run a Tau-shaped synthetic throughput sweep against the Modal server."""

    load_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    server = serve_no_eager if no_eager else serve
    base_url = server.get_web_url()
    print(f"Modal vLLM URL: {base_url}")
    print(
        "load sweep:",
        {
            "variant": variant,
            "no_eager": no_eager,
            "concurrency_values": concurrency_values,
            "requests_per_level": requests_per_level,
            "max_tokens": max_tokens,
        },
    )

    health_started = time.perf_counter()
    wait_for_health(base_url, timeout_seconds=STARTUP_TIMEOUT_SECONDS)
    startup_to_health = time.perf_counter() - load_started
    print(f"health: ok after {startup_to_health:.1f}s")

    # Warm the route and tokenizer before measuring concurrency levels.
    warmup = asyncio.run(
        run_load_level(
            base_url=base_url,
            concurrency=1,
            requests=2,
            max_tokens=32,
            timeout=timeout,
            request_offset=0,
        )
    )
    print("warmup:", json.dumps(warmup, indent=2)[:2000])

    levels: list[dict[str, Any]] = []
    request_offset = 10_000
    for raw_value in concurrency_values.split(","):
        concurrency = int(raw_value.strip())
        if concurrency <= 0:
            continue
        requests = max(requests_per_level, concurrency * 2)
        print(f"running level: concurrency={concurrency}, requests={requests}")
        level = asyncio.run(
            run_load_level(
                base_url=base_url,
                concurrency=concurrency,
                requests=requests,
                max_tokens=max_tokens,
                timeout=timeout,
                request_offset=request_offset,
            )
        )
        request_offset += requests + 1
        levels.append(level)
        print("level_result:", json.dumps(level, indent=2)[:3000])

    result = {
        "variant": variant,
        "started_at": started_at,
        "model_id": MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "gpu_type": GPU_TYPE,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "kv_cache_memory_bytes": None,
        "enforce_eager": not no_eager,
        "enable_prefix_caching": ENABLE_PREFIX_CACHING,
        "skip_mm_profiling": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "generation_config": "vllm",
        "load_format": "safetensors",
        "safetensors_load_strategy": "eager",
        "max_tokens": max_tokens,
        "requests_per_level": requests_per_level,
        "startup_to_health_seconds": startup_to_health,
        "health_wait_seconds": time.perf_counter() - health_started,
        "warmup": warmup,
        "levels": levels,
        "total_experiment_seconds": time.perf_counter() - load_started,
    }
    result_dir = (
        Path(__file__).resolve().parents[1]
        / "20260512-qwen35-2b-throughput"
        / "results"
    )
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"_{variant}.json"
    )
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_load_summary(result, result_path)
    print("wrote:", result_path)
    print("wrote:", result_path.with_suffix(".md"))


@app.local_entrypoint()
def smoke() -> None:
    """Spin up the Modal server, run tiny API checks, then let it scale down."""

    smoke_started = time.perf_counter()
    timings: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "served_model_name": SERVED_MODEL_NAME,
        "gpu_type": GPU_TYPE,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "kv_cache_memory_bytes": None,
        "enforce_eager": True,
        "skip_mm_profiling": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0},
        "optimization_level": None,
        "generation_config": "vllm",
        "load_format": "safetensors",
        "safetensors_load_strategy": "eager",
    }

    base_url = serve.get_web_url()
    timings["web_url_seconds"] = time.perf_counter() - smoke_started
    print(f"Modal vLLM URL: {base_url}")
    health_started = time.perf_counter()
    wait_for_health(base_url, timeout_seconds=STARTUP_TIMEOUT_SECONDS)
    timings["startup_to_health_seconds"] = time.perf_counter() - smoke_started
    timings["health_wait_seconds"] = time.perf_counter() - health_started
    print("health: ok")

    models_started = time.perf_counter()
    models = get_json(f"{base_url}/v1/models")
    timings["models_request_seconds"] = time.perf_counter() - models_started
    print("models:", json.dumps(models, indent=2)[:1200])

    chat_payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ],
        "temperature": 0.0,
        "max_tokens": 64,
    }
    chat_started = time.perf_counter()
    chat = post_json(f"{base_url}/v1/chat/completions", chat_payload)
    timings["chat_request_seconds"] = time.perf_counter() - chat_started
    timings["chat_usage"] = chat.get("usage")
    print("chat:", json.dumps(chat, indent=2)[:2000])

    tool_payload = {
        "model": SERVED_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the weather tool for Paris. Do not answer directly."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name, for example Paris.",
                            }
                        },
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "temperature": 0.0,
        "max_tokens": 128,
    }
    try:
        tool_started = time.perf_counter()
        tool = post_json(f"{base_url}/v1/chat/completions", tool_payload)
        timings["tool_request_seconds"] = time.perf_counter() - tool_started
        timings["tool_usage"] = tool.get("usage")
        timings["tool_calls"] = (
            ((tool.get("choices") or [{}])[0].get("message") or {}).get("tool_calls")
            or []
        )
        print("tool_call:", json.dumps(tool, indent=2)[:3000])
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print("tool_call_failed:", exc.code, body[:3000])
        raise

    timings["total_smoke_seconds"] = time.perf_counter() - smoke_started
    result_dir = Path(__file__).resolve().parent / "smoke_results"
    result_dir.mkdir(exist_ok=True)
    result_path = result_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_t4_eager_skip_mm_default_opt_auto_kv.json"
    )
    result_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    (result_dir / "latest.json").write_text(
        json.dumps(timings, indent=2), encoding="utf-8"
    )
    print("timings:", json.dumps(timings, indent=2))
    print(f"wrote timings: {result_path}")
    print("smoke: ok")
