# Tau Custom Harness Wiring Notes

Do not run the benchmark from this folder yet. This is only the wiring note for
the next step after the vLLM endpoint is healthy.

The custom banking harness already goes through LiteLLM. To point it at this
Modal/vLLM endpoint later, use the hosted-vLLM provider shape:

```bash
uv run python custom_harness/tau3_custom_harness/run_banking.py \
  --task-id task_006 \
  --agent-model hosted_vllm/qwen3.5-2b \
  --user-model hosted_vllm/qwen3.5-2b \
  --subagent-model hosted_vllm/qwen3.5-2b \
  --agent-llm-args-json '{"api_base":"https://YOUR_MODAL_URL/v1","api_key":"EMPTY","max_tokens":512}' \
  --user-llm-args-json '{"api_base":"https://YOUR_MODAL_URL/v1","api_key":"EMPTY","max_tokens":512}' \
  --subagent-llm-args-json '{"api_base":"https://YOUR_MODAL_URL/v1","api_key":"EMPTY","max_tokens":512}' \
  --max-steps 20 \
  --timeout 300
```

Keep the first harness probe short. The first real thing to measure is not score;
it is whether Qwen3.5-2B emits usable OpenAI-style tool calls through vLLM.
