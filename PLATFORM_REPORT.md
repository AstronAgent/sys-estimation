# Qwen3.6-35B-A3B LLM Platform — System Requirements & Cost Report

**Prepared for:** Client
**Date:** 2026-08-20
**Version:** 1.1 — supersedes 1.0. Serving hardware moved from H100 (`p5.48xlarge`) to **H200 (`p5e.48xlarge`)**; costs, latency budget, and the training-hardware answer all follow from that.
**Companion files:**
- `SYSTEM_REQUIREMENTS.md` — engineering detail
- `deployment-architecture.md` — Fig. 1 request path and Fig. 2 training pipeline as Mermaid (renders on GitHub/GitLab)
- `deployment-architecture.html` — the same figures as inline SVG, with PNG/PDF export

---

## Executive summary

A self-hosted platform serving **Qwen/Qwen3.6-35B-A3B** on SGLang to 3,000 daily users, fronted by a graph-based MCP harness, with authenticated chat history, a vector database, a continuously-ingested knowledge graph, and a continuous RL pipeline with hot-swap deployment.

**The whole platform fits on one AWS instance.**

| | |
|---|---|
| **Recommended infrastructure** | 1× `p5e.48xlarge` (8× H200 141 GB) + ~$6k/mo supporting services |
| **All 8 GPUs allocated** | 2 live serving · 1 standby · 1 canary · 4 training |
| **Annual cost (3-yr reserved)** | **$311,089** |
| **Annual cost (on-demand)** | $626,029 |
| **Cost per conversation** | **$0.028** (3-yr reserved) / $0.057 (on-demand) |
| **Cost per user per month** | **$8.64** (3-yr reserved) / $17.39 (on-demand) |
| **Modelled response time** | ~4.6 s end-to-end, inside the 2–8 s SLA with ~3.4 s of slack |

### Five findings that shaped the design

1. **The model is Mixture-of-Experts (35B total, ~3B active).** VRAM is sized by total parameters, but compute and bandwidth by active parameters. Decode runs ~4–5× faster than a dense 35B and each replica fits on **one** GPU instead of two. This halved the GPU fleet from 6 to 3.

2. **H200 rather than H100, and the reason is the SLA — not the memory.** H200 is the same compute die with 141 GB of HBM3e at 4.8 TB/s instead of 80 GB at 3.35 TB/s. Decode is bandwidth-bound, so it runs **~1.43× faster**: ~4.6 s end-to-end instead of ~6.4 s, and ~255 KV slots per replica instead of ~96. On a 3-year commitment that costs **$31,247/year more — about 15%**. See §10.2.

3. **Concurrent sessions are not GPU streams.** By Little's Law only **~25 sessions are actually generating** at peak, out of 750 open. Sizing GPUs against the session count would over-provision by an order of magnitude — which is why revising the session target from 2,000 down to 750 does not change the GPU bill at all.

4. **Latency, not throughput, sets the replica count — and H200 changes what that means.** One replica clears peak token throughput, and on H200 one replica also *meets* the 8 s ceiling (~6.7 s). The second active replica now buys margin rather than compliance, and single-replica operation becomes a genuine degraded mode during maintenance.

5. **The 6-iteration cap no longer breaches the SLA — but it is close.** On H100 the worst case ran ~9.6 s; on H200 it runs **~7.0 s**, inside the ceiling with only ~1 s to spare. That is not a guarantee. An iteration cap does not bound latency — a **wall-clock deadline** does. See §5.1.

### Two decisions required before procurement

- **Confirm "pre-training" means domain-adaptive continued pre-training**, not training a base model from scratch. From-scratch is not feasible on the proposed hardware — see §7.2.
- **Confirm self-hosting is being chosen for capability, not cost.** At this volume a commercial API would be roughly 4× cheaper. The premium buys model ownership and the RL loop — which the requirements describe as core. See §9.

---

## 1. Requirements as specified

| Environment | Purpose |
|---|---|
| 1. Pre-training | Base / continued pre-training |
| 2. Post-training (RL) | Continuous RLHF / RLAIF |
| 3. Production Testing | Canary + eval gate before promotion |
| 4. Async Live Production | Customer-facing inference, **~750 concurrent sessions** |

| Parameter | Value |
|---|---|
| Model | Qwen/Qwen3.6-35B-A3B (MoE) |
| Serving runtime | SGLang |
| Daily active users | 3,000 |
| Conversations per user per day | 10 |
| Peak concurrent sessions | **~750** (revised down from 2,000) |
| Tokens per conversation | 5,000 |
| LLM passes per conversation | 2–3 typical, 6 hard cap |
| Response time SLA | **2–8 seconds** |
| Cloud | AWS |
| RL / pre-training hardware | 1× H200 or 1× B200 |
| Dedicated ML model | 16 GB VRAM / 64 GB RAM / 16 vCPU, runs on a fixed frequency, results cached |
| Knowledge graph | Full graph, continuous ingestion, +8 vCPU for weight calculation |

---

## 2. Load model

**Agentic amplification.** Each loop iteration re-sends the accumulated context, so prefill grows quadratically with iteration count while decode grows linearly. Modelling a typical 3-pass conversation ending at 5,000 tokens:

| Pass | Input (cumulative) | Output |
|---|---|---|
| 1 — user turn → tool call | 1,800 | 150 |
| 2 — + tool result → tool call | 3,400 | 150 |
| 3 — + tool result → final answer | 5,000 | 700 |
| **Prefill, uncached** | **10,200** | |
| **Prefill, with prefix cache** | **~5,000** | |
| **Decode** | | **1,000** |

SGLang's RadixAttention prefix caching removes roughly **half the total prefill work**. The inference load balancer must route by prefix hash so successive passes land on the same replica. This is load-bearing, not an optimisation.

| Metric | Per day | Average | Peak (3.5×) |
|---|---|---|---|
| Conversations | 30,000 | 0.35/s | **1.2/s** |
| LLM inference calls | 75,000 | 0.87/s | ~3/s |
| **Decode tokens** | **30 M** | 350 tok/s | **~1,200 tok/s** |
| Prefill tokens (cached) | 150 M | 1,750 tok/s | ~6,100 tok/s |

**Concurrency.** Decode wall-time per conversation ≈ 5 s; peak arrival ≈ 1.2/s. Little's Law gives **~25 concurrent decoding streams** at peak, against 750 open sessions. Sessions are cheap (Redis); decode slots are expensive (HBM).

> **~25 is retained as a conservative upper bound.** Two separate effects push the true figure lower. Faster decode on H200 drains the queue faster. And Little's Law applied literally to the arrival rate above gives ~6, not ~25 — a ~4× gap that is not derived anywhere in this report, and which most likely reflects a business-hours load profile that was never written down. The sizing below deliberately takes neither credit: it holds concurrency at ~25 so the batch sizes in §4.1 stay pessimistic, because fewer streams means smaller batches, faster per-stream decode, and better latency than modelled. **The full model and this open item are set out in `SYSTEM_REQUIREMENTS.md` §2.1 and §2.1.8**, and both are on the pre-sign-off checklist.

> **The complete mathematical model — symbols, equations, and the five constraints the design must satisfy — is stated explicitly in `SYSTEM_REQUIREMENTS.md` §2.1.** Every figure in this report is reproducible from it plus the assumptions in Appendix A.

> **The session count does not drive GPU sizing.** Little's Law runs off arrival rate, not open sessions, so revising the target from 2,000 to 750 leaves the GPU fleet unchanged. It reduces the Redis working set (~400 MB → ~150 MB) and trims the Chat API autoscale range — about $119/month in total.
>
> **750 is also the more internally consistent figure.** At 1.2 conversations/s peak arrival, 750 concurrent sessions implies an average session of ~10.4 minutes — plausible for a chat with 2–6 agentic turns. The earlier 2,000 would have implied ~28-minute sessions.

---

## 3. GPU sizing

### 3.1 What A3B means

| Dimension | Governed by | Value |
|---|---|---|
| VRAM for weights | **Total** params (35B) | ~35 GB at FP8 |
| Compute per token | **Active** params (3B) | ~11× less than dense 35B |
| Memory bandwidth per token | **Active** params (3B) | ~3 GB read/token at FP8 |
| KV cache | Layers × KV heads | ~48 KB/token at FP8 |

### 3.2 Precision decides the topology

| Precision | Weights | KV headroom on 1× H200 141 GB | Verdict |
|---|---|---|---|
| BF16 | ~70 GB | ~63 GB → ~164 streams | Now *fits* at TP=1 on 141 GB — but reads 2× the bytes per token, so decode roughly halves. Rejected on bandwidth, not memory. |
| **FP8** | **~35 GB** | **~98 GB → ~255 streams @ 8K ctx** | **Recommended. TP=1, one GPU per replica.** |
| INT4 | ~18 GB | ~115 GB → ~300 streams | Cost tier; benchmark quality first |

```
KV/token = 2 (K+V) × 48 layers × 4 kv_heads × 128 head_dim × 1 byte (FP8)
         = 49,152 B = 48 KiB/token  →  8K context ≈ 384 MB/stream
Capacity  = (141 − 35 weights − 8 overhead) ÷ 0.384 ≈ 255 streams per replica
```

> **The extra 61 GB is not the reason to buy H200 — the extra bandwidth is.** 96 KV slots per replica already exceeded the ~25 concurrent streams of peak demand by roughly 4×, so H100 was never memory-constrained at FP8. What H200 buys is 4.8 TB/s instead of 3.35 TB/s, and decode is bandwidth-bound. The KV headroom is a bonus — it matters for long-context growth and for co-resident training, not for today's load.

*Assumes a Qwen3-30B-A3B-class config. Confirm layer count, KV heads and head_dim against the actual model card.*

### 3.3 Throughput per replica (1× H200, FP8, TP=1)

| Metric | H100 80 GB | **H200 141 GB** | Scaling |
|---|---|---|---|
| Single-stream decode | ~200–250 tok/s | **~290–360 tok/s** | bandwidth, ×1.43 |
| Per-stream decode at batch ~12 | ~150–180 tok/s | **~215–260 tok/s** | bandwidth, ×1.43 |
| Per-stream decode at batch ~25 | ~110–130 tok/s | **~160–185 tok/s** | bandwidth, ×1.43 |
| Aggregate decode | ~2,000–4,000 tok/s | **~2,900–5,700 tok/s** | bandwidth, ×1.43 |
| Prefill | ~20,000–30,000 tok/s | **~20,000–30,000 tok/s** | **unchanged** |

> **Prefill does not speed up, and that is expected.** H200 carries the same GH100 die as H100 — identical SM count, identical FP8 throughput (~1,979 TFLOPS peak). Only the memory subsystem changed. Prefill is compute-bound and sees no gain; decode is bandwidth-bound and sees the full 4.8 ÷ 3.35 = **1.43×**. Every latency improvement in §4 comes from the decode terms alone.

Peak decode demand is ~1,200 tok/s. One replica clears it with a wide margin. Throughput does not set the replica count.

---

## 4. Latency budget — the binding constraint

With the dedicated ML model moved off the request path (it runs on a schedule and writes to cache), a 3-pass conversation:

| Step | H100 | **H200** |
|---|---|---|
| Pass 1 prefill (1,800 tok @ ~25k tok/s) | 0.07 s | 0.07 s |
| Pass 1 decode (150 tok @ 180 → 260 tok/s) | 0.83 s | **0.58 s** |
| Tool round 1 | 0.15 s | 0.15 s |
| Pass 2 prefill (+1,600, prefix cached) | 0.06 s | 0.06 s |
| Pass 2 decode (150 tok) | 0.83 s | **0.58 s** |
| Tool round 2 | 0.15 s | 0.15 s |
| Pass 3 prefill (+1,600) | 0.06 s | 0.06 s |
| **Pass 3 decode (700 tok — final answer)** | 3.9 s | **2.69 s** |
| Orchestrator + network | 0.30 s | 0.30 s |
| **End-to-end** | ~6.4 s | **~4.6 s** |

**~3.4 s of slack against the 8 s ceiling, up from ~1.6 s on H100.** Time-to-first-token on the final answer drops to **~1.9 s**, from ~2.6 s.

### 4.1 Replica count follows from the SLA

| Config | Batch/replica | Per-stream decode | End-to-end | SLA |
|---|---|---|---|---|
| 1 active | ~25 | ~170 tok/s | **~6.7 s** | ✓ meets — thin margin |
| **2 active** | ~12 | ~260 tok/s | **~4.6 s** | ✓ **recommended** |
| 3 active | ~8 | ~290 tok/s | ~4.2 s | ✓ marginal gain |

**The recommendation is unchanged — 2 active + 1 standby = 3 GPUs — but the reasoning has shifted.** On H100 the second active replica was *required* to meet the SLA. On H200 a single replica already lands at ~6.7 s, so the second one buys p99 margin rather than compliance, and it costs nothing marginal because all 8 GPUs sit on the one instance already.

The real gain is operational: **single-replica serving is now a supported degraded mode.** The fleet can lose a replica, or drain one for a weight swap, and still answer inside the 2–8 s window. That was not true on H100.

The standby covers N+1 failover and gives hot-swap a staging target without a capacity dip.

---

## 5. Architecture

See `deployment-architecture.md` (Mermaid) or `deployment-architecture.html` (SVG, exportable), Fig. 1.

```
Users → ALB + WAF → Chat API (+ Cognito auth, + Redis sessions)
      → SQS → MCP Graph Orchestrator
                 ├─ Inference LB → SGLang pool (2 active + 1 standby)   [agentic loop, 2–6 passes]
                 └─ MCP tool servers: Postgres · Vector DB · Knowledge Graph · ML result cache
```

**Two load balancers, deliberately different:**

| LB | Placement | Algorithm | Rationale |
|---|---|---|---|
| ALB | Internet → Chat API | Round-robin | TLS, WAF, rate limiting. ~3 req/s — trivial. |
| **Inference LB** | Orchestrator → SGLang | **Least-outstanding + prefix-hash affinity** | LLM request durations vary 10×+, so least-outstanding beats round-robin. Prefix affinity is what makes the cache hit across loop iterations. |

Round-robin in front of an LLM pool is a common and costly mistake — it ignores both duration variance and cache locality.

### 5.1 Required guards

Three, all necessary:

1. **Iteration cap (6)** — bounds tool-call depth.
2. **Per-conversation token budget** — the iteration cap alone does not bound cost, because one pass can carry a very large context.
3. **Wall-clock deadline (6 s)** — at 6 s elapsed, stop issuing tool calls and force an answer from what has been gathered. **This is the only hard guarantee against the 8-second ceiling.**

> **H200 narrows this problem but does not remove it.** On H100 the 6-iteration worst case ran ~9.6 s — an outright breach. On H200 it runs **~7.0 s**: inside the ceiling, but with about one second of margin that a slow tool call or a prefix-cache miss will consume. Keep the deadline. It is now a safety net rather than the only thing holding the SLA together, and that is a materially better position to operate from.

Additionally: **stream the final answer over SSE** (time-to-first-token ~1.9 s on H200, so the user sees output well inside the window), and **dispatch independent tools in parallel** — that is what the graph edges are for, and each saved round is ~0.8 s.

### 5.2 Security

- **Two auth layers.** User-facing: OIDC/OAuth2, JWT with refresh, per-user rate limits. Service-to-service: **per-tool scoped tokens**.
- The LLM must never hold one credential reaching every backend. A prompt-injected tool call would otherwise reach the entire data tier.
- GPU subnet private, no public ingress; orchestrator is the only client of the inference pool.

---

## 6. Data tier

At 3,000 DAU the data tier is small. Do not over-provision — but keep a horizontal path open.

| Store | Purpose | Growth |
|---|---|---|
| RDS PostgreSQL | Users, auth, chat history | ~600 MB/day → **~220 GB/yr**, ~5 QPS. No sharding needed. |
| Vector DB | RAG + semantic memory | ~600 MB/day → **~500 GB/yr** with index overhead |
| Knowledge Graph | Full graph, continuous ingestion | ~300k nodes + 600k edges/day → **~110 M nodes/yr, ~45 GB/yr** |
| DynamoDB | Scheduled ML model results | Small; native TTL |
| S3 | Checkpoints | 35 GB per FP8 checkpoint; 50 retained ≈ **2 TB** |

### 6.1 Knowledge graph — two cautions

> **8 vCPU supports incremental weight calculation, not global.** At ~12 writes/second, incremental or windowed edge-weight updates are comfortable on 8 cores. Recomputing a global metric (PageRank, centrality, community detection) over a 110-million-node graph is a distributed-compute job — at year-two scale it would run for hours. If global weights are required, schedule them as periodic batch jobs on separate compute (Neptune Analytics or Spark on EMR), or approximate them incrementally.

> **A continuously ingesting graph with no retention policy grows without bound.** Add decay, pruning, or entity consolidation from day one. Retrofitting it onto a 200-million-edge graph is painful.

---

## 7. Training pipeline

```
Pre-training → Post-training (RL) → S3 Model Registry
    → Production Testing (canary + shadow traffic)
    → Automated Eval Gate
    → promote → hot-swap into the live SGLang fleet
                     ↳ production traces + reward signal feed back into RL
```

### 7.1 RL post-training on one GPU

| Approach | Policy | Reference | Grads | Optimizer | Activations | **Total** | H200 141 GB | B200 180 GB |
|---|---|---|---|---|---|---|---|---|
| Full-param BF16 + AdamW FP32 | 70 | 70 | 70 | 280 | ~30 | **~520 GB** | ✗ | ✗ |
| Full-param FP8 + 8-bit Adam | 35 | 35 | 35 | 70 | ~30 | **~205 GB** | ✗ | ✗ |
| **LoRA on FP8 base** | 35 | **0** | ~1 | ~4 | ~25 | **~66 GB** | **✓** | **✓** |
| QLoRA (NF4 base) | 18 | **0** | ~1 | ~4 | ~25 | **~48 GB** | ✓ | ✓ |

Two terms drive that table:

- **AdamW FP32 optimizer state is 8 bytes per trainable parameter** — 280 GB on 35B. It is the largest single term and the reason full-parameter RL fits on no single GPU that exists today.
- **With LoRA the reference model is free.** The KL-penalty reference is the base model with adapters disabled — same weights, no second copy.

**Verdict: LoRA RL fits comfortably on one H200 or B200. Full-parameter RL does not.**

> **The H200 serving decision resolves the training-hardware question outright.** The requirement asked for *1× H200 or 1× B200* for RL and pre-training. AWS sells neither as a single GPU — both ship only as 8-GPU instances. Because the serving fleet is now a `p5e.48xlarge`, **four spare H200s are already in the plan**, and the requirement is met exactly as written, on hardware that is already paid for. No separate training instance, and no substitution of an 80 GB card for the 141 GB one the client specified.

*MoE guidance:* apply LoRA to attention projections (q, k, v, o). Adapting all expert FFNs is expensive and rarely necessary for alignment. The router can be adapted but destabilises easily — start without it.

*Throughput (RL is rollout-generation-bound):* **~600–1,200 GRPO steps/day on H200**, ~1,000–2,000 on B200. Sufficient for continuous incremental RL from production traces; not for a from-scratch alignment campaign.

### 7.2 Pre-training on one GPU

| Scenario | Memory | Time on 1 GPU | Verdict |
|---|---|---|---|
| **From scratch**, ~2T tokens | ~560 GB | **~2.9 yr (H200) / ~1.3 yr (B200)** | **Not feasible** |
| Full-parameter continued PT | ~560 GB | — | **Not feasible on memory** |
| **LoRA continued / domain-adaptive** | ~66 GB | see below | **Viable** |

```
FLOPs/token ≈ 6 × 3e9 active params = 1.8e10
H200 @ ~400 TFLOPS effective (40% MFU) → ~22,000 tok/s → ~1.9 B tokens/day
B200 @ ~900 TFLOPS effective          → ~50,000 tok/s → ~4.3 B tokens/day
```

**A 10–50 B token domain corpus takes ~5–26 days on H200, or ~2–12 days on B200.** That is a genuinely useful capability and almost certainly what is intended.

> **Decision required:** confirm "pre-training" means domain-adaptive continued pre-training. Training a base model from scratch needs hundreds of GPUs and a fundamentally different budget.

### 7.3 Hot-swap and the promotion gate

- **Preferred:** SGLang's live weight-update path (`update_weights_from_disk` / distributed weight sync). Weights replaced in-place in a running server; connections stay open, no cold start.
- **Fallback:** blue/green drain through the inference load balancer with automatic rollback.
- **Gate:** every checkpoint clears an automated eval suite — safety, regression, task accuracy, **p99 latency against the 8 s SLA**, tool-call validity — before reaching live traffic.

> **There must be no direct path from the RL loop into the live environment.** A continuous training pipeline that can push straight to production is a continuous outage pipeline.

---

## 8. Cost breakdown

**Pricing basis:** AWS us-east-1, Linux, August 2026. The GPU on-demand rate is verified against published sources (see Sources); **the 3-year reserved rate is derived, not quoted — see the note in §8.1.** Non-GPU rates are list-price estimates and should be confirmed in the AWS Pricing Calculator before a purchase order. Prices exclude taxes, support plans, and engineering effort. AWS raised H200 instance rates ~15% during 2025, so re-verify at purchase rather than relying on this document.

### 8.1 GPU tier — one instance, fully allocated

`p5e.48xlarge` — 8× NVIDIA H200 141 GB HBM3e (4.8 TB/s), 192 vCPU, 2,048 GiB RAM. `p5en.48xlarge` is the same GPUs on an Emerald Rapids host with EFAv3 networking; it prices slightly differently and is the SKU with the most reliable published rate.

| GPUs | Allocation |
|---|---|
| 2 | Live SGLang — active |
| 1 | Live SGLang — standby (N+1 + hot-swap staging) |
| 1 | Production Testing canary |
| 4 | LoRA RL + continued pre-training, scheduled off-peak |
| **8** | **Fully allocated — nothing idle** |

| Commitment | $/hour | $/month (730 h) | $/year |
|---|---|---|---|
| On-demand | $63.296 | $46,206 | **$554,473** |
| 1-year Savings Plan *(est. ~30% off — confirm with AWS)* | ~$44.31 | ~$32,344 | ~$388,131 |
| **3-year Reserved** *(derived — see note)* | **~$27.344** | **$19,961** | **$239,533** |

**The 3-year commitment saves ~57% — $314,940/year.** This is by far the largest single lever in the entire budget.

> **The 3-year rate is derived, not quoted, and it is the number to verify first.** AWS publishes a verified 3-year reserved rate for `p5.48xlarge` ($23.777/hr, which is 43.2% of its $55.040 on-demand rate) but does not publish an equivalent open rate for `p5e`. The $27.344/hr above applies that same 43.2% ratio to the p5e on-demand rate. **Confirm it directly with AWS or through the Pricing Calculator before issuing a purchase order** — it carries roughly $240k/year, and a materially worse discount ratio would change the H200-vs-H100 comparison in §10.2.

*An idle GPU on this instance costs ~$7.91/hour on-demand. The allocation above exists so none sits idle.*

### 8.2 Non-GPU compute

| Component | Instance | ~$/hr | Qty | $/month |
|---|---|---|---|---|
| MCP Graph Orchestrator | `m7g.2xlarge` | 0.326 | 3 | $714 |
| Chat API (autoscaled 2–4) | `m7g.xlarge` | 0.163 | 2 avg | $238 |
| Auth service | `m7g.large` | 0.082 | 2 | $120 |
| Graph ingestion worker (8 vCPU) | `c7g.2xlarge` | 0.290 | 1 | $212 |
| Vector DB (self-hosted) | `r7g.xlarge` | 0.214 | 3 | $469 |
| Scheduled ML model | `g6.4xlarge` | 1.323 | ~4 h/day | $159 |
| | | | **Subtotal** | **~$1,912** |

### 8.3 Managed services and storage

| Service | Configuration | $/month |
|---|---|---|
| RDS PostgreSQL | `db.r7g.2xlarge` Multi-AZ + 1 TB gp3 | ~$1,530 |
| Neptune (knowledge graph) | `db.r6g.2xlarge` + read replica | ~$1,700 |
| ElastiCache Redis | `cache.r7g.large` × 2 (~150 MB working set) | ~$330 |
| CloudWatch + AMP + AMG | logs, metrics, OTel traces | ~$300 |
| S3 | 5 TB standard (checkpoints, artifacts) | ~$115 |
| Application Load Balancer | 1 ALB + LCU | ~$30 |
| DynamoDB | ML result cache, on-demand, TTL | ~$25 |
| Data transfer out | ~120 MB/day + overhead | ~$20 |
| SQS | 75k messages/day | ~$1 |
| | **Subtotal** | **~$4,051** |

### 8.4 Totals

| | Monthly | Annual |
|---|---|---|
| **On-demand** | | |
| GPU | $46,206 | $554,473 |
| Non-GPU compute | $1,912 | $22,944 |
| Managed services | $4,051 | $48,612 |
| **Total** | **$52,169** | **$626,029** |
| **1-year Savings Plan** *(GPU only)* | | |
| **Total** | **~$38,307** | **~$459,687** |
| **3-year Reserved** *(GPU only)* | | |
| GPU | $19,961 | $239,533 |
| Non-GPU compute | $1,912 | $22,944 |
| Managed services | $4,051 | $48,612 |
| **Total** | **$25,924** | **$311,089** |

*Non-GPU compute and managed services are unaffected by the GPU choice — the entire H100 → H200 delta lands in the GPU line.*

### 8.5 Unit economics

| Metric | On-demand | 1-yr SP | **3-yr Reserved** |
|---|---|---|---|
| Cost per conversation *(10.95 M/yr)* | $0.0572 | $0.0420 | **$0.0284** |
| Cost per user per month *(3,000 DAU)* | $17.39 | $12.77 | **$8.64** |
| Cost per 1 M decode tokens *(10.95 B/yr)* | $57.17 | $41.98 | **$28.41** |

*H100 equivalents for comparison: $0.0256 per conversation and $7.77 per user per month on a 3-year commit. The H200 premium is **$0.0028 per conversation** — about a quarter of a cent — in exchange for ~1.8 s of response time.*

### 8.6 Three-year total cost of ownership

| | 3-yr Reserved |
|---|---|
| GPU | $718,599 |
| Non-GPU compute | $68,832 |
| Managed services | $145,836 |
| **Infrastructure TCO** | **$933,267** |

*On H100 the equivalent three-year figure is $839,526. The H200 decision adds **$93,741 over three years**.*

> **Infrastructure is only part of TCO.** Building and operating this platform — the MCP graph harness, the RL pipeline, the eval gate, the ingestion workers, and ongoing on-call — is a substantial engineering commitment that is likely comparable to or larger than the infrastructure line. It is deliberately not priced here because it depends on team composition, but it should not be omitted from the business case.

---

## 9. Build vs. buy — an honest comparison

At 10,200 input + 1,000 output tokens per conversation, and representative commercial API pricing of ~$0.50 per million input / ~$1.50 per million output tokens:

```
Per conversation: (10,200 × $0.0000005) + (1,000 × $0.0000015) = $0.0066
Annual (10.95 M conversations):                                  ≈ $72,000
```

| | Annual |
|---|---|
| Commercial API | **~$72,000** |
| Self-hosted on H200 (3-yr reserved) | **$311,089** |
| Self-hosted on H100 (3-yr reserved) | $279,842 |
| **Premium for self-hosting** | **~4.3×** (H200) / ~3.9× (H100) |

**At this volume, self-hosting is not the cheaper option.** The break-even is roughly **129,000 conversations/day — about 4.3× current volume** — because GPU cost is fixed while API cost scales linearly.

> **The H100-vs-H200 choice barely moves this comparison.** Both sit in the same 4× band against the API. If the build-vs-buy premium is acceptable at 3.9× it is acceptable at 4.3×, and if it is not acceptable at 4.3× then downgrading to H100 does not rescue the case — the decision to self-host has to stand on the RL loop, not on GPU selection.

Self-hosting is nonetheless justified here, for reasons that are not cost:

- **Continuous RL ownership.** A commercial API model cannot be RL-tuned on your production traces. The requirements make this loop central to the product — this alone is the strongest argument.
- **Data residency and privacy.** Conversations and the knowledge graph never leave your VPC.
- **Deterministic latency control.** You own the tail against a hard 2–8 s SLA.
- **Deep MCP and graph integration** without per-call vendor constraints.
- **No per-token vendor exposure** as volume grows — and the economics invert above ~129k conversations/day.

> **Recommendation:** proceed with self-hosting if the RL loop is genuinely core to the product. If it is not — if the model would in practice run static — the API path is materially cheaper and the platform should be reconsidered. This is a product decision, not an engineering one, and it should be made explicitly rather than by default.
>
> *API pricing varies widely by model tier; the comparison shifts with model choice. Re-run it against the specific alternative you would actually use.*

---

## 10. Cost optimisation levers, ranked

| # | Lever | Saving | Notes |
|---|---|---|---|
| 1 | **3-year Reserved / Savings Plan** | **~$315k/yr (57%)** | Largest lever by an order of magnitude. Requires a 3-year commitment, and the p5e reserved rate must be confirmed with AWS (§8.1). |
| 2 | **Keep all 8 GPUs allocated** | ~$7.91/GPU/hr avoided | Already in the plan — training uses the spare 4 rather than a second instance. |
| 3 | Self-host graph DB instead of Neptune | ~$1,200/mo | Neo4j on `r7g.2xlarge` ≈ $500/mo vs Neptune ≈ $1,700. Adds operational burden. |
| 4 | INT4 quantisation | Possibly 1 replica | Halves weight memory, more KV headroom. Quality tax must be benchmarked first. |
| 5 | Parallel tool dispatch | GPU-seconds + latency | Fewer passes per conversation cuts both cost and response time. |
| 6 | Aggressive prefix caching | ~50% of prefill | Already core to the design; protect it with prefix-affinity routing. |
| 7 | Canary only during release windows | Frees 1 GPU | No direct saving (same instance) but adds training capacity. |
| 8 | **Downgrade to H100 (`p5.48xlarge`)** | **~$31k/yr (10%)** | Real, but it spends the SLA margin the H200 was bought for. See §10.2 before taking it. |
| 9 | Spot for RL / continued pre-training | Situational | Only helps if training moves off the p5e — but AWS sells no single H200, so a separate instance would cost far more than it saves. |

### 10.1 Options considered and rejected

| Option | Why rejected |
|---|---|
| L40S (`g6e`) for serving | 864 GB/s bandwidth vs H200's 4.8 TB/s → ~40–50 tok/s per stream → ~14–17 s responses. **Breaks the SLA by a wide margin.** |
| A100 80 GB (`p4de`) for serving | 2.0 TB/s → ~110 tok/s per stream → ~9 s end-to-end. **Breaches.** Cheaper but does not meet 2–8 s. |
| Separate H200/B200 training instance | Now unnecessary as well as uneconomic: the serving instance *is* an 8× H200 machine with 4 GPUs spare. Buying a second `p5e.48xlarge` to use one GPU would roughly double the GPU bill. |
| BF16 serving | On H200 this now fits — 70 GB weights leaves ~63 GB for KV, ~164 streams at TP=1, where on H100 it left ~2 GB and was impossible. It is still rejected, but on **bandwidth**: BF16 reads twice the bytes per token, cutting decode to ~130 tok/s at batch 12 and pushing end-to-end back to ~8.5 s. FP8 is the correct precision on H200 for speed, not for capacity. |
| B200 (`p6-b200.48xlarge`) for serving | ~8.0 TB/s would decode ~1.7× faster again, but the SLA is already met with ~3.4 s of slack at a materially lower rate. Worth revisiting only if context lengths or pass counts grow substantially. |

### 10.2 The H200 decision, and the H100 downgrade option

`p5e` (H200) is the recommendation. `p5` (H100) remains a legitimate cheaper alternative, so the comparison is set out in full:

| | H100 (`p5.48xlarge`) | **H200 (`p5e` / `p5en.48xlarge`)** |
|---|---|---|
| VRAM / bandwidth per GPU | 80 GB HBM3, 3.35 TB/s | **141 GB HBM3e, 4.8 TB/s** |
| Compute (FP8 peak) | ~1,979 TFLOPS | ~1,979 TFLOPS — **identical die** |
| KV headroom after 35 GB weights | ~37 GB → ~96 slots | **~98 GB → ~255 slots** |
| Decode speed (bandwidth-proportional) | 1.0× | **~1.43×** |
| Per-stream decode at batch 12 | ~180 tok/s | **~260 tok/s** |
| Modelled end-to-end (2 active) | ~6.4 s | **~4.6 s** |
| Slack against the 8 s ceiling | ~1.6 s | **~3.4 s** |
| 6-iteration worst case | ~9.6 s — **breaches** | **~7.0 s — inside** |
| Single-replica degraded mode | ~8.5 s — breaches | **~6.7 s — holds** |
| RL / pre-training GPU | 80 GB — below the 141 GB specified | **141 GB — exactly as specified** |
| On-demand | $55.040/hr → $482,150/yr | $63.296/hr → **$554,473/yr** |
| 3-year reserved | $23.777/hr → $208,286/yr | ~$27.344/hr → **$239,533/yr** *(derived)* |
| **Annual delta, 3-yr reserved** | — | **+$31,247 (+15%)** |

**What the 15% buys, concretely:**

- **The 6-iteration worst case stops being an SLA breach.** On H100 it was ~9.6 s and the wall-clock deadline was the only thing preventing a violation. On H200 it lands at ~7.0 s on its own.
- **Losing a replica stops being an SLA event.** Single-replica operation holds at ~6.7 s, so maintenance, weight swaps, and a node failure no longer force a choice between capacity and latency.
- **Headroom against the assumptions that are least certain.** If the prefix-cache hit rate lands below the modelled ~50%, or the iteration distribution is fatter-tailed than assumed, H200 absorbs it inside the existing fleet. On H100 either would have required a third active replica.
- **The training requirement is met as written.** The client specified 1× H200; the spare GPUs are H200s. On `p5`, "1× H200" would have been silently substituted with an 80 GB H100.

**When to take the H100 downgrade instead:** if $31k/year is material to the business case, *and* the 6-iteration tail is measured to be rare, *and* the wall-clock deadline is implemented and tested. It is a defensible trade — but it should be a deliberate one, made after benchmarking, not a default.

> **Verify the p5e reserved rate before treating the 15% figure as firm** (§8.1). It is derived from p5's discount ratio, not quoted by AWS. If the real p5e reserved discount is worse, the delta widens and this comparison should be re-run.

---

## 11. Risks and open items

### Decisions required before procurement

1. **Confirm "pre-training" scope** (§7.2). From-scratch is not feasible; domain-adaptive continued pre-training is.
2. **Confirm the build-vs-buy premium is accepted** (§9), and that the RL loop justifies it.
3. **Region and capacity strategy.** `p5e` availability is thinner than `p5` in several regions. **Securing H200 supply is a lead-time risk, not only a cost question** — evaluate EC2 Capacity Blocks for ML early, and confirm p5e or p5en availability in the target region before the design depends on it.
4. **Confirm the `p5e` 3-year reserved rate with AWS** (§8.1). It is derived rather than quoted and carries ~$240k/year.

### Needed to finalise sizing

5. **Exact Qwen3.6-35B-A3B config** — layer count, KV heads, head_dim, expert count. The KV math in §3.2 assumes a Qwen3-30B-A3B-class shape.
6. **Scheduled ML model:** run frequency, maximum acceptable staleness, and cache-miss behaviour (serve stale / trigger on-demand / return unavailable — the last is usually correct for a latency-bound system).
7. **Knowledge-graph extraction density** — entities and edges per conversation, which drives §6 growth numbers.
8. **Whether graph weight calculation is incremental or global** — 8 vCPU supports the former, not the latter at scale.

### Engineering risks

| Risk | Impact | Mitigation |
|---|---|---|
| Six-iteration conversations approach the 8 s SLA (~7.0 s on H200) | Thin tail margin | Wall-clock deadline (§5.1), SSE streaming, parallel tool dispatch |
| Prefix-cache hit rate below the assumed ~50% | Prefill load doubles | Measure early; H200's decode margin and 4 spare GPUs absorb it on the same instance |
| Knowledge graph grows unbounded | Query latency degrades, storage climbs | Retention/decay policy from day one (§6.1) |
| Training job starves a serving replica | Latency breach | Pin GPUs explicitly; schedule training off-peak. Note training now shares 141 GB cards, so a LoRA job at ~66 GB leaves real headroom rather than filling the device |
| Prompt injection reaches backend tools | Data exposure | Per-tool scoped tokens; the LLM never holds a superuser credential (§5.2) |
| RL checkpoint regresses production | Quality incident | Automated eval gate with p99 latency check; no direct RL→prod path (§7.3) |
| `p5e` capacity unavailable in the target region | Procurement delay | Confirm p5e/p5en availability early; `p5` (H100) remains a working fallback at the SLA cost set out in §10.2 |
| p5e 3-year reserved rate worse than the derived $27.344/hr | Budget overrun on the largest line | Confirm with AWS before the purchase order (§8.1) |
| Concurrency figure (~25 streams) not derived from the stated arrival rate | Batch sizes, and therefore every latency number, rest on an unverified input | Conservative direction — real latency should be better, not worse. Measure directly under production traffic; see `SYSTEM_REQUIREMENTS.md` §2.1.8 |

---

## 12. Recommendations

1. **Procure one `p5e.48xlarge` (8× H200 141 GB) on a 3-year Reserved commitment** — ~$239,533/year, versus $554,473 on-demand. Allocate all 8 GPUs: 2 live, 1 standby, 1 canary, 4 training.
2. **Confirm the p5e 3-year reserved rate with AWS before signing** (§8.1). It is derived from p5's discount ratio and carries ~$240k/year. Confirm `p5e`/`p5en` capacity in the target region at the same time.
3. **Treat `p5` (H100) as the costed fallback, not the default** (§10.2). It saves $31,247/year and gives back ~1.8 s of response time, the 6-iteration margin, and single-replica degraded mode.
4. **Serve at FP8, TP=1.** FP8 remains correct on H200 for bandwidth reasons, not memory ones — BF16 now fits on 141 GB but decodes roughly half as fast. Benchmark INT4 separately as a cost tier.
5. **Implement all three guards** — iteration cap, token budget, and wall-clock deadline. H200 pulls the 6-iteration worst case inside the ceiling, but ~1 s of margin is not a guarantee.
6. **Stream over SSE** and **dispatch independent tools in parallel**. Both cut perceived latency and GPU-seconds.
7. **Use least-outstanding-requests with prefix-hash affinity** at the inference LB. Not round-robin.
8. **Train with LoRA on the spare H200s**, off-peak. This meets the stated "1× H200" requirement exactly, with no separate instance. Full-parameter RL is not viable on one GPU and does not need to be.
9. **Gate every promotion** on an automated eval suite including p99 latency.
10. **Set a knowledge-graph retention policy before go-live**, not after.
11. **Make the build-vs-buy decision explicitly** (§9) before capital is committed.

---

## 13. Validate before sign-off

- [ ] Benchmark Qwen3.6-35B-A3B on a single H200 at FP8 — confirm ~290–360 tok/s single-stream decode and ~255 KV slots at 8K context
- [ ] **Confirm the ~1.43× decode scaling from H100 empirically.** The entire latency case rests on decode being bandwidth-bound; if the measured gain is materially below 1.43×, re-run §4 and §10.2
- [ ] **Measure p99 end-to-end latency under peak concurrency against the 8 s ceiling** — this is the acceptance criterion, not throughput
- [ ] Measure the 6-iteration worst case specifically — modelled at ~7.0 s with ~1 s of margin
- [ ] Measure the real prefix-cache hit rate under agentic traffic (assumed ~50% prefill saving)
- [ ] **Measure concurrent decoding streams and decode wall-time under production-shaped traffic** — resolves the ~6-vs-~25 gap (`SYSTEM_REQUIREMENTS.md` §2.1.8) that underpins every batch size and latency figure
- [ ] **Confirm the daily load profile** — 24-hour uniform, or concentrated into business hours
- [ ] Measure the actual distribution of iterations per conversation (assumed avg 2.5, cap 6)
- [ ] Confirm SSE streaming delivers time-to-first-token under ~3 s
- [ ] Load-test graph ingestion at 12 writes/s with weight calculation running concurrently on 8 vCPU
- [ ] Rehearse a hot-swap under live traffic — confirm zero dropped connections
- [ ] Rehearse an eval-gate failure — confirm automatic rollback
- [ ] Re-confirm all pricing in the AWS Pricing Calculator for the target region on the day of purchase

---

## Appendix A — Assumptions

Every number in this report derives from the inputs in §1 plus the following. Each is stated so it can be challenged and re-run.

| # | Assumption | Value | Impact if wrong |
|---|---|---|---|
| 1 | Peak factor over daily average | 3.5× | Linear on peak throughput and replica count |
| 2 | Average LLM passes per conversation | 2.5 | Linear on decode volume |
| 3 | Output split across passes | 150 / 150 / 700 tokens | Shifts the latency budget |
| 4 | Prefix-cache saving on prefill | ~50% | Below this, prefill load rises; a 3rd active replica may be needed |
| 5 | Per-stream decode at batch 12 (H200) | ~260 tok/s | Directly sets end-to-end latency |
| 5b | H200 decode scaling over H100 | ×1.43 (4.8 ÷ 3.35 TB/s) | Whole latency case in §4 and the H200 premium in §10.2 |
| 6 | Model config | Qwen3-30B-A3B-class (48 layers, 4 KV heads, head_dim 128) | Changes KV/token and slot count |
| 7 | Decode wall-time per conversation | ~5 s | Sets concurrent stream count via Little's Law |
| 8 | Knowledge-graph extraction density | ~10 entities + 20 edges per conversation | Linear on graph growth and storage |
| 9 | Training MFU | 40% | Linear on continued-pre-training throughput |
| 10 | Scheduled ML model runtime | ~4 h/day | Linear on that line item only (~$159/mo) |
| 11 | Commercial API comparison rate | $0.50/M in, $1.50/M out | Shifts the build-vs-buy multiple in §9 |
| 12 | Utilisation basis | 730 hours/month | Standard AWS month |
| 13 | `p5e` 3-year reserved discount | 43.2% of on-demand, taken from `p5` | ±$25k/yr on the largest line; **derived, not quoted — confirm with AWS** |
| 14 | Concurrent decoding streams at peak | ~25 (conservative) | Sets batch size and therefore per-stream decode. **Does not follow from Appx 1 + 7, which give ~6** — see `SYSTEM_REQUIREMENTS.md` §2.1.8. Conservative, so latency is better than modelled, not worse |
| 15 | Daily load profile | 24-hour uniform | The concurrency figure in use implies business-hours concentration instead. Restating it moves peak arrival, not the daily totals |

**All figures are back-of-the-envelope estimates.** They are directionally correct and sufficient to scope a purchase order, but the GPU count and the latency SLA must be confirmed by load-testing the actual model on the actual hardware before procurement.

---

## Sources

Pricing verified August 2026, us-east-1:

**Recommended instance (H200):**

- [p5en.48xlarge pricing and specs — CloudPrice](https://cloudprice.net/aws/ec2/instances/p5en.48xlarge) — 8× H200 141 GB, **~$63.296/hr on-demand** — the rate used throughout this report
- [p5e.48xlarge pricing and specs — Vantage](https://instances.vantage.sh/aws/ec2/p5e.48xlarge) — listed on-demand figure appears inconsistent with its own spot rate; **confirm p5e pricing directly with AWS**
- [AWS quietly increases prices for H200 EC2 instances by 15% — Data Center Dynamics](https://www.datacenterdynamics.com/en/news/aws-quietly-increases-prices-for-h200-ec2-instances-by-15/) — H200 rates have moved recently; re-verify at purchase
- **No open published 3-year reserved rate for `p5e` / `p5en` was found.** The $27.344/hr used in §8.1 is derived from p5's verified discount ratio and must be confirmed with AWS.

**H100 fallback, and the basis for the derived reserved rate:**

- [p5.48xlarge pricing and specs — Vantage](https://instances.vantage.sh/aws/ec2/p5.48xlarge) — $55.040/hr on-demand, $23.777/hr 3-year reserved (43.2% — the ratio applied to p5e)
- [p5.48xlarge Specs & Pricing — DoiT Compute](https://www.doit.com/compute/spot/us-east-1/p5.48xlarge)

**Considered and rejected:**

- [p6-b200.48xlarge Specs & Pricing — DoiT Compute](https://www.doit.com/compute/spot/us-east-1/p6-b200.48xlarge) — $113.9328/hr on-demand
- [Prerequisites for Capacity Blocks — AWS documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-prerequisites.html)

Non-GPU instance and managed-service rates in §8.2 and §8.3 are list-price estimates and are **not** individually verified. Confirm them in the AWS Pricing Calculator for the target region before issuing a purchase order.
