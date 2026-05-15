# Qwen3.5-2B Modal T4 Startup Results

All runs used:

- GPU: `T4`
- model: `Qwen/Qwen3.5-2B`
- served name: `qwen3.5-2b`
- context: `8192`
- `--enforce-eager`
- native tool-call smoke request

| variant | startup to health | total smoke | first chat | tool call | notes |
|---|---:|---:|---:|---:|---|
| text-only / skip-mm | 604.4s | 619.2s | 11.06s | 2.55s | Saved memory, but startup got worse. |
| O0 + safetensors + text-only | 441.7s | 452.9s | 7.78s | 2.23s | Biggest startup win so far. |
| O0 + safetensors + text-only + 4GiB KV | 417.8s | 428.4s | 7.37s | 2.15s | Best current startup; vLLM skipped automatic KV memory profiling. |
| default opt + safetensors + text-only + auto KV | 502.9s | 516.5s | 9.87s | 2.57s | Benchmark-oriented variant after removing `--optimization-level 0` and fixed KV. Auto KV gave 7.81 GiB cache and 66x max concurrency at 8192 tokens; engine init took 404.93s. |

Current best command tail:

```bash
--max-model-len 8192 \
--gpu-memory-utilization 0.88 \
--kv-cache-memory-bytes 4294967296 \
--enforce-eager \
--skip-mm-profiling \
--limit-mm-per-prompt '{"image":0,"video":0}' \
--optimization-level 0 \
--generation-config vllm \
--load-format safetensors \
--safetensors-load-strategy eager
```

The current best still supports OpenAI-style tool calls. The smoke tool request
returned:

```json
{"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}
```

The latest Modal app was verified stopped after the smoke run.

Benchmark-oriented command tail after undoing the startup-biased knobs:

```bash
--max-model-len 8192 \
--gpu-memory-utilization 0.88 \
--enforce-eager \
--skip-mm-profiling \
--limit-mm-per-prompt '{"image":0,"video":0}' \
--generation-config vllm \
--load-format safetensors \
--safetensors-load-strategy eager
```
