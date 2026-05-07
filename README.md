# continual-agentic-rl

Proof of work for continual RL on production-style agents.

The idea is simple:

1. Evaluate a strong big model on Tau3 banking.
2. Use it to generate high-quality agent traces.
3. Distill those traces into a smaller open model.
4. Run RL on the small model.
5. Show that it improves on held-out production-style tasks.

Today, most agent teams improve systems by fixing prompts, harnesses, tools, retrieval, and policies. Then they create corrected traces and SFT on them. That loop helps, but eventually it saturates.

We want to show that production traces can become RL data, so agents can keep improving after prompt fixes and SFT stop moving the metric.

## Starting Point

We are starting with Tau3 banking because it has the shape of real agent work:

- multi-turn users
- tool calls
- hidden database state
- policies and knowledge-base constraints
- binary rewards

Explorer:

https://tau3-banking-explorer.vercel.app/

## What This Repo Is For

This repo will hold the code, traces, eval configs, and training scripts for the proof-of-work loop:

**big model eval -> distillation -> small model RL -> held-out eval**

## Team

Ayush, Manan, Vyom, [Kanishk](https://github.com/kanishkez).
