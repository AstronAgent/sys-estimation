# Qwen3.6-35B-A3B — self-hosted LLM platform, AWS cost estimate

Server sizing and AWS cost for a self-hosted **Qwen/Qwen3.6-35B-A3B** platform: SGLang serving, a graph-based MCP harness, a continuously-ingested knowledge graph, and a continuous RL pipeline with gated hot-swap deployment.

**3,000 daily users · 30,000 conversations/day · 2–8 second response SLA.**

---

## The number

| | 3-year reserved | 1-year savings plan | On-demand |
|---|---|---|---|
| **Annual, all-in** | **$311,089** | ~$459,687 | $626,029 |
| Monthly | $25,924 | ~$38,307 | $52,169 |
| Per conversation | **$0.0284** | $0.0420 | $0.0572 |
| Per user per month | **$8.64** | $12.77 | $17.39 |

**The whole platform runs on one AWS instance** — a `p5e.48xlarge` (8× NVIDIA H200 141 GB), with all 8 GPUs allocated and none idle:

| GPUs | Role |
|---|---|
| 2 | Live serving — active |
| 1 | Live serving — standby (N+1 and hot-swap staging) |
| 1 | Production testing canary |
| 4 | LoRA RL and continued pre-training, scheduled off-peak |

About **$6k/month** of supporting services sits on top of the GPU line — database, vector store, knowledge graph, cache, load balancer, monitoring. Full breakdown in [PLATFORM_REPORT.md §8](PLATFORM_REPORT.md).

> **The 3-year commitment is the largest single lever in the budget — ~$315,000/year, about 57%.** Nothing else in the analysis comes close.

---

## What the money buys

### Fig. 1 — Live production request path

![Live production request path: users → ALB/WAF → Chat API → SQS → MCP Graph Orchestrator → inference LB → 3× SGLang replicas on H200, plus the MCP tool tier and offline jobs](images/fig1-request-path.png)

Two active SGLang replicas plus one standby, each a single H200 141 GB at FP8/TP=1, behind a least-outstanding-requests load balancer with prefix-hash affinity. The orchestrator holds the agentic loop; the GPUs only decode.

### Fig. 2 — Training, gated hot-swap, and the scheduled ML track

![Training pipeline: LoRA pre-training → GRPO RL on a spare H200 → S3 model registry → canary production testing → automated eval gate → hot-swap into the live fleet, with production traces feeding back into RL](images/fig2-training-hotswap.png)

RL and continued pre-training run on the spare H200s of the same instance. Nothing reaches live traffic without clearing an automated eval gate — including a p99 latency check against the 8-second SLA.

---

## What drives the cost

- **A3B is Mixture-of-Experts, and that halves the fleet.** VRAM is sized by *total* parameters (35B), but compute and memory bandwidth by *active* parameters (~3B). Decode runs ~4–5× faster than a dense 35B, so each replica fits on one GPU instead of two.
- **H200 over H100 costs ~15% more and buys the SLA back.** 141 GB HBM3e at 4.8 TB/s versus 80 GB at 3.35 TB/s gives ~1.43× decode — ~4.6 s end-to-end instead of ~6.4 s, and ~255 KV slots per replica instead of ~96. On a 3-year commit the delta is **$31,247/year**. H100 remains a costed fallback; the full comparison is in §10.2 of the report.
- **Latency sets the replica count, not throughput.** One replica clears peak tokens/second roughly 2.4× over. The 8-second ceiling is what the fleet is actually sized against.
- **Concurrent sessions are not GPU streams.** Only a few percent of the ~750 open sessions are generating at any instant. Sizing GPUs against the session count would over-provision by an order of magnitude.
- **Self-hosting is ~4.3× more expensive than a commercial API at this volume** (~$72k/year). It is justified by the RL loop, data residency, and latency ownership — **not by cost**. Break-even is around 129,000 conversations/day. This is a product decision and should be made explicitly; see §9 of the report.

---

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
| Cloud | AWS (`us-east-1`) |

---

## Contents

| File | What it is |
|---|---|
| [PLATFORM_REPORT.md](PLATFORM_REPORT.md) | **Start here.** Full report: sizing, latency budget, complete AWS cost breakdown, build-vs-buy, risks, recommendations. |
| [deployment-architecture.md](deployment-architecture.md) | Both diagrams — rendered images plus the Mermaid source. |
| [deployment-architecture.html](deployment-architecture.html) | Same diagrams as self-contained inline SVG, with PNG/PDF export buttons. Open locally. |
| [images/](images/) | Rendered figures — PNG for viewing, SVG for print. |
| [reference/SYSTEM_REQUIREMENTS.md](reference/SYSTEM_REQUIREMENTS.md) | Derivation and assumptions behind every figure — load model, the explicit mathematical model (§2.1), GPU sizing, subsystem specs, and the single-GPU training assessment. **Optional reading.** |

---

## Confidence and caveats

These are **directional back-of-the-envelope estimates** — enough to scope a purchase order, not a substitute for benchmarking. Every assumption is listed in Appendix A of the report so it can be challenged and re-run. Three things should be settled before capital is committed:

1. **Confirm the `p5e` 3-year reserved rate with AWS.** The on-demand rate ($63.296/hr) is published; the 3-year rate used here ($27.344/hr) is **derived** by applying `p5`'s verified 43.2% discount ratio, because AWS does not publish a p5e reserved rate openly. It carries roughly $240k/year.
2. **Benchmark the real model on a single H200** and measure p99 against the 8-second ceiling. Confirm decode scales ~1.43× over H100 — the latency case rests on it.
3. **Confirm the daily load profile.** The concurrency figure used for batch sizing implies traffic concentrated into business hours rather than spread evenly over 24 hours — documented in [reference/SYSTEM_REQUIREMENTS.md §2.1.8](reference/SYSTEM_REQUIREMENTS.md). It errs conservative, so real latency should be better than modelled, but it should be measured rather than assumed.

AWS pricing verified August 2026 for `us-east-1`. Re-confirm in the AWS Pricing Calculator on the day of purchase — AWS raised H200 instance rates ~15% during 2025.
