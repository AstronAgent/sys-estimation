# sys-estimation

System requirements, architecture, and AWS cost estimation for a self-hosted **Qwen/Qwen3.6-35B-A3B** LLM platform — SGLang serving, a graph-based MCP harness, a continuously-ingested knowledge graph, and a continuous RL pipeline with gated hot-swap deployment.

## Contents

| File | What it is |
|---|---|
| [PLATFORM_REPORT.md](PLATFORM_REPORT.md) | **Start here.** Client-facing report: sizing, latency budget, full AWS cost breakdown, build-vs-buy analysis, risks, recommendations. |
| [SYSTEM_REQUIREMENTS.md](SYSTEM_REQUIREMENTS.md) | Engineering detail — load model, GPU sizing derivation, subsystem specs, single-GPU training assessment. |
| [deployment-architecture.md](deployment-architecture.md) | Architecture diagrams as Mermaid (renders inline on GitHub). |
| [deployment-architecture.html](deployment-architecture.html) | Same diagrams as self-contained inline SVG, with PNG/PDF export buttons. Open locally. |
| [deployment-architecture-artifact.html](deployment-architecture-artifact.html) | Publish-safe copy — no CDN dependencies, no export toolbar. |

## Design target

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3.6-35B-A3B (MoE — 35B total, ~3B active) |
| Serving | SGLang, FP8, TP=1 |
| Daily active users | 3,000 |
| Conversations per day | 30,000 (10 per user) |
| Tokens per conversation | 5,000 |
| Agentic passes per conversation | 2–3 typical, 6 hard cap |
| Peak concurrent sessions | ~750 |
| Response time SLA | 2–8 seconds |
| Cloud | AWS |

## Headline conclusions

- **The whole platform fits on one `p5.48xlarge`** (8× H100 80 GB), with all 8 GPUs allocated: 2 live serving, 1 standby, 1 canary, 4 training.
- **$279,842/year** on a 3-year reserved commitment — **$0.0256 per conversation**, **$7.77 per user per month**. On-demand is $553,706/year, so the 3-year commit is worth ~57%.
- **A3B is Mixture-of-Experts, and that halves the fleet.** VRAM is sized by total parameters, compute and bandwidth by active parameters — so decode runs ~4–5× faster than a dense 35B and each replica fits on one H100 instead of two.
- **Concurrent sessions are not GPU streams.** By Little's Law only ~25 sessions are generating at peak, out of ~750 open. Sizing GPUs against the session count would over-provision by an order of magnitude.
- **Latency sets the replica count, not throughput.** One replica clears peak tokens/second; the 8-second ceiling is what requires two.
- **The 6-iteration cap breaches the SLA** (~9.6 s). An iteration cap does not bound latency — a wall-clock deadline does.
- **LoRA training fits on one GPU; full-parameter RL does not** (~66 GB vs ~520 GB — AdamW FP32 optimizer state alone is 280 GB). Pre-training a base model from scratch would take 1.3–2.9 years on a single GPU and is not feasible.
- **Self-hosting is ~3.9× more expensive than a commercial API at this volume.** It is justified by the RL loop, data residency, and latency ownership — not by cost. Break-even is around 116,000 conversations/day.

## Status

Directional back-of-the-envelope estimates, sufficient to scope a purchase order. Every assumption is listed in Appendix A of the report. **Benchmark the real model on a single H100 and measure p99 against the 8-second ceiling before procurement.** AWS pricing verified August 2026 for `us-east-1`; re-confirm in the AWS Pricing Calculator on the day of purchase.
