# Qwen3.5-2B vLLM Modal Smoke

Goal: self-host `Qwen/Qwen3.5-2B` on a Modal GPU through vLLM, test the
OpenAI-compatible API, and shut it down. This is not a benchmark run.

## Files

- `modal_vllm_server.py`: Modal app that starts `vllm serve`.
- `harness/smoke_client.py`: small standalone client for a running endpoint.
- `harness/tau_custom_harness_notes.md`: how to point the custom Tau3 harness at
  the endpoint later.

## Smoke Test

```bash
modal run experiments/20260512-qwen35-2b-vllm-modal/modal_vllm_server.py
```

The local entrypoint checks:

1. `/health`
2. `/v1/models`
3. `/v1/chat/completions` with a normal chat request
4. `/v1/chat/completions` with a simple tool-call request

The Modal function uses `scaledown_window=60`, so the GPU should stop roughly a
minute after the smoke test stops sending requests. You can also force-stop it:

```bash
modal app stop qwen35-2b-vllm-tau3
```

## First Settings

- GPU: `T4`
- Max model length: `8192`
- vLLM image: `vllm/vllm-openai:v0.20.2`
- Tool parser: `qwen3_xml`

The tool parser choice comes from the model chat template: Qwen3.5 emits XML
tool calls with `<tool_call>` and `<function=...>` blocks.

For smoke testing, the server uses `--enforce-eager` and `max_containers=1`.
That keeps first boot cheap and prevents Modal from spinning up extra T4 cold
starts while the first container is still loading.

The startup smoke also disables multimodal profiling for our text-only banking
harness:

```bash
--skip-mm-profiling
--limit-mm-per-prompt '{"image":0,"video":0}'
```

Each smoke run writes timing JSON under `smoke_results/`.
