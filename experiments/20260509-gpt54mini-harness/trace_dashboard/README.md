# Tau3 Trace Dashboard

Static dashboard for inspecting one custom-harness banking trace.

Build the current task 006 data:

```bash
uv run python experiments/20260509-gpt54mini-harness/trace_dashboard/build_trace_data.py
```

Serve it locally:

```bash
cd experiments/20260509-gpt54mini-harness/trace_dashboard
python3 -m http.server 8791
```

Open:

```text
http://127.0.0.1:8791/
```

The view has three parts:

- `Agent Tree`: planner calls with KB subagent runs nested under them.
- `Timeline`: public user, agent, environment, and tool messages.
- `Failure`: compact notes on why the task failed.
