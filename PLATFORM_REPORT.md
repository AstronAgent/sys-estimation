# Qwen3.6-35B-A3B LLM Platform — System Requirements & Cost Report

**Prepared for:** Client
**Date:** 2026-08-20
**Version:** 1.0
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
| **Recommended infrastructure** | 1× `p5.48xlarge` (8× H100 80 GB) + ~$6k/mo supporting services |
| **All 8 GPUs allocated** | 2 live serving · 1 standby · 1 canary · 4 training |
| **Annual cost (3-yr reserved)** | **$279,842** |
| **Annual cost (on-demand)** | $553,706 |
| **Cost per conversation** | **$0.026** (3-yr reserved) / $0.051 (on-demand) |
| **Cost per user per month** | **$7.77** (3-yr reserved) / $15.38 (on-demand) |
| **Modelled response time** | ~6.4 s end-to-end, inside the 2–8 s SLA |

### Four findings that shaped the design

1. **The model is Mixture-of-Experts (35B total, ~3B active).** VRAM is sized by total parameters, but compute and bandwidth by active parameters. Decode runs ~4–5× faster than a dense 35B and each replica fits on **one** H100 instead of two. This halved the GPU fleet from 6 to 3.

2. **Concurrent sessions are not GPU streams.** By Little's Law only **~25 sessions are actually generating** at peak, out of 750 open. Sizing GPUs against the session count would over-provision by an order of magnitude — which is why revising the session target from 2,000 down to 750 does not change the GPU bill at all.

3. **Latency, not throughput, sets the replica count.** One replica clears peak token throughput. The 2–8 second SLA is what requires two active replicas.

4. **The 6-iteration cap breaches the SLA.** A conversation that hits the cap runs ~9.6 s. An iteration cap does not bound latency — a **wall-clock deadline** does. See §5.1.

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

| Precision | Weights | KV headroom on 1× H100 80 GB | Verdict |
|---|---|---|---|
| BF16 | ~70 GB | ~2 GB — unusable | Would force TP=2 purely for memory |
| **FP8** | **~35 GB** | **~37 GB → ~96 streams @ 8K ctx** | **Recommended. TP=1, one GPU per replica.** |
| INT4 | ~18 GB | ~54 GB | Cost tier; benchmark quality first |

```
KV/token = 2 (K+V) × 48 layers × 4 kv_heads × 128 head_dim × 1 byte (FP8)
         = 49,152 B = 48 KiB/token  →  8K context ≈ 384 MB/stream
Capacity  = (80 − 35 weights − 8 overhead) ÷ 0.384 ≈ 96 streams per replica
```

*Assumes a Qwen3-30B-A3B-class config. Confirm layer count, KV heads and head_dim against the actual model card.*

### 3.3 Throughput per replica (1× H100, FP8, TP=1)

| Metric | Estimate |
|---|---|
| Single-stream decode | ~200–250 tok/s |
| Per-stream decode at batch ~12 | ~150–180 tok/s |
| Per-stream decode at batch ~25 | ~110–130 tok/s |
| Aggregate decode | ~2,000–4,000 tok/s |
| Prefill | ~20,000–30,000 tok/s |

Peak decode demand is ~1,200 tok/s. One replica clears it. Throughput does not set the replica count.

---

## 4. Latency budget — the binding constraint

With the dedicated ML model moved off the request path (it runs on a schedule and writes to cache), a 3-pass conversation:

| Step | Time |
|---|---|
| Pass 1 prefill (1,800 tok @ ~25k tok/s) | 0.07 s |
| Pass 1 decode (150 tok @ 180 tok/s) | 0.83 s |
| Tool round 1 | 0.15 s |
| Pass 2 prefill (+1,600, prefix cached) | 0.06 s |
| Pass 2 decode (150 tok) | 0.83 s |
| Tool round 2 | 0.15 s |
| Pass 3 prefill (+1,600) | 0.06 s |
| **Pass 3 decode (700 tok — final answer)** | **3.9 s** |
| Orchestrator + network | 0.30 s |
| **End-to-end** | **~6.4 s** |

### 4.1 Replica count follows from the SLA

| Config | Batch/replica | Per-stream decode | End-to-end | SLA |
|---|---|---|---|---|
| 1 active | ~25 | ~120 tok/s | **~8.5 s** | ✗ breaches |
| **2 active** | ~12 | ~180 tok/s | **~6.4 s** | ✓ meets |
| 3 active | ~8 | ~200 tok/s | ~6.0 s | ✓ marginal gain |

**2 active + 1 standby = 3 GPUs.** The standby covers N+1 failover and gives hot-swap a staging target without a capacity dip.

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
3. **Wall-clock deadline (6 s)** — at 6 s elapsed, stop issuing tool calls and force an answer from what has been gathered. **This is the only hard guarantee against the 8-second ceiling.** Six iterations run ~9.6 s and would otherwise breach it.

Additionally: **stream the final answer over SSE** (time-to-first-token ~2.6 s, so the user sees output well inside the window), and **dispatch independent tools in parallel** — that is what the graph edges are for, and each saved round is ~1 s.

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

**Verdict: LoRA RL fits comfortably on one H200, B200, or H100 80 GB. Full-parameter RL does not.**

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

**Pricing basis:** AWS us-east-1, Linux, August 2026. GPU instance rates verified against published sources (see Sources). Non-GPU rates are list-price estimates and should be confirmed in the AWS Pricing Calculator before a purchase order. Prices exclude taxes, support plans, and engineering effort.

### 8.1 GPU tier — one instance, fully allocated

`p5.48xlarge` — 8× NVIDIA H100 80 GB, 192 vCPU, 2,048 GiB RAM.

| GPUs | Allocation |
|---|---|
| 2 | Live SGLang — active |
| 1 | Live SGLang — standby (N+1 + hot-swap staging) |
| 1 | Production Testing canary |
| 4 | LoRA RL + continued pre-training, scheduled off-peak |
| **8** | **Fully allocated — nothing idle** |

| Commitment | $/hour | $/month (730 h) | $/year |
|---|---|---|---|
| On-demand | $55.040 | $40,179 | **$482,150** |
| 1-year Savings Plan *(est. ~30% off — confirm with AWS)* | ~$38.53 | ~$28,127 | ~$337,505 |
| **3-year Reserved** | **$23.777** | **$17,357** | **$208,286** |

**The 3-year commitment saves ~57% — $273,864/year.** This is by far the largest single lever in the entire budget.

*An idle GPU on this instance costs ~$6.88/hour on-demand. The allocation above exists so none sits idle.*

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
| GPU | $40,179 | $482,150 |
| Non-GPU compute | $1,912 | $22,944 |
| Managed services | $4,051 | $48,612 |
| **Total** | **$46,142** | **$553,706** |
| **1-year Savings Plan** *(GPU only)* | | |
| **Total** | **~$34,090** | **~$409,081** |
| **3-year Reserved** *(GPU only)* | | |
| GPU | $17,357 | $208,286 |
| Non-GPU compute | $1,912 | $22,944 |
| Managed services | $4,051 | $48,612 |
| **Total** | **$23,320** | **$279,842** |

### 8.5 Unit economics

| Metric | On-demand | 1-yr SP | **3-yr Reserved** |
|---|---|---|---|
| Cost per conversation *(10.95 M/yr)* | $0.0506 | $0.0374 | **$0.0256** |
| Cost per user per month *(3,000 DAU)* | $15.38 | $11.36 | **$7.77** |
| Cost per 1 M decode tokens *(10.95 B/yr)* | $50.57 | $37.36 | **$25.56** |

### 8.6 Three-year total cost of ownership

| | 3-yr Reserved |
|---|---|
| GPU | $624,858 |
| Non-GPU compute | $68,832 |
| Managed services | $145,836 |
| **Infrastructure TCO** | **$839,526** |

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
| Self-hosted (3-yr reserved) | **$279,842** |
| **Premium for self-hosting** | **~3.9×** |

**At this volume, self-hosting is not the cheaper option.** The break-even is roughly **116,000 conversations/day — about 3.9× current volume** — because GPU cost is fixed while API cost scales linearly.

Self-hosting is nonetheless justified here, for reasons that are not cost:

- **Continuous RL ownership.** A commercial API model cannot be RL-tuned on your production traces. The requirements make this loop central to the product — this alone is the strongest argument.
- **Data residency and privacy.** Conversations and the knowledge graph never leave your VPC.
- **Deterministic latency control.** You own the tail against a hard 2–8 s SLA.
- **Deep MCP and graph integration** without per-call vendor constraints.
- **No per-token vendor exposure** as volume grows — and the economics invert above ~117k conversations/day.

> **Recommendation:** proceed with self-hosting if the RL loop is genuinely core to the product. If it is not — if the model would in practice run static — the API path is materially cheaper and the platform should be reconsidered. This is a product decision, not an engineering one, and it should be made explicitly rather than by default.
>
> *API pricing varies widely by model tier; the comparison shifts with model choice. Re-run it against the specific alternative you would actually use.*

---

## 10. Cost optimisation levers, ranked

| # | Lever | Saving | Notes |
|---|---|---|---|
| 1 | **3-year Reserved / Savings Plan** | **~$274k/yr (57%)** | Largest lever by an order of magnitude. Requires a 3-year commitment. |
| 2 | **Keep all 8 GPUs allocated** | ~$6.88/GPU/hr avoided | Already in the plan — training uses the spare 4 rather than a second instance. |
| 3 | Self-host graph DB instead of Neptune | ~$1,200/mo | Neo4j on `r7g.2xlarge` ≈ $500/mo vs Neptune ≈ $1,700. Adds operational burden. |
| 4 | INT4 quantisation | Possibly 1 replica | Halves weight memory, more KV headroom. Quality tax must be benchmarked first. |
| 5 | Parallel tool dispatch | GPU-seconds + latency | Fewer passes per conversation cuts both cost and response time. |
| 6 | Aggressive prefix caching | ~50% of prefill | Already core to the design; protect it with prefix-affinity routing. |
| 7 | Canary only during release windows | Frees 1 GPU | No direct saving (same instance) but adds training capacity. |
| 8 | Spot for RL / continued pre-training | Situational | Only helps if training moves off the p5 — but AWS sells no single H100, so a separate instance may cost more than it saves. |

### 10.1 Options considered and rejected

| Option | Why rejected |
|---|---|
| L40S (`g6e`) for serving | 864 GB/s bandwidth vs H100's 3.35 TB/s → ~40–50 tok/s per stream → ~14–17 s responses. **Breaks the SLA.** |
| A100 80 GB (`p4de`) for serving | 2.0 TB/s → ~110 tok/s per stream → ~9 s end-to-end. **Marginal to breaching.** Cheaper but does not meet 2–8 s. |
| Separate H200/B200 training instance | AWS sells only 8-GPU SKUs (`p5e.48xlarge`, `p6-b200.48xlarge`). Buying a second 8-GPU machine to use one GPU costs far more than using the 4 already spare. |
| BF16 serving | 70 GB weights leaves ~2 GB for KV on an 80 GB card. Unusable without TP=2, which doubles the GPU count for no gain. |

### 10.2 Worth pricing: H200 for serving

| | H100 (`p5.48xlarge`) | H200 (`p5e` / `p5en.48xlarge`) |
|---|---|---|
| KV headroom after 35 GB weights | ~37 GB → **~96 slots** | ~98 GB → **~255 slots** |
| Decode speed (bandwidth-proportional) | 1.0× | **~1.43×** |
| Per-stream decode at batch 12 | ~180 tok/s | **~260 tok/s** |
| Modelled end-to-end | ~6.4 s | **~5.2 s** |
| On-demand rate | $55.04/hr | ~$63.30/hr *(p5en; confirm p5e with AWS)* |

For roughly **15% more**, H200 buys ~1.2 s of additional SLA margin, 2.6× the KV capacity, and far more headroom for the RL and continued-pre-training workloads sharing the box. It may also permit dropping to 1 active + 1 standby. **Worth pricing directly against p5 before committing.**

---

## 11. Risks and open items

### Decisions required before procurement

1. **Confirm "pre-training" scope** (§7.2). From-scratch is not feasible; domain-adaptive continued pre-training is.
2. **Confirm the build-vs-buy premium is accepted** (§9), and that the RL loop justifies it.
3. **Region and capacity strategy.** p5 availability varies by region. **Securing H100 supply is a lead-time risk, not only a cost question** — evaluate EC2 Capacity Blocks for ML early.

### Needed to finalise sizing

4. **Exact Qwen3.6-35B-A3B config** — layer count, KV heads, head_dim, expert count. The KV math in §3.2 assumes a Qwen3-30B-A3B-class shape.
5. **Scheduled ML model:** run frequency, maximum acceptable staleness, and cache-miss behaviour (serve stale / trigger on-demand / return unavailable — the last is usually correct for a latency-bound system).
6. **Knowledge-graph extraction density** — entities and edges per conversation, which drives §6 growth numbers.
7. **Whether graph weight calculation is incremental or global** — 8 vCPU supports the former, not the latter at scale.

### Engineering risks

| Risk | Impact | Mitigation |
|---|---|---|
| Six-iteration conversations breach the 8 s SLA | SLA violation on the tail | Wall-clock deadline (§5.1), SSE streaming, parallel tool dispatch |
| Prefix-cache hit rate below the assumed ~50% | Prefill load doubles; may need a 3rd active replica | Measure early; capacity exists on the same instance |
| Knowledge graph grows unbounded | Query latency degrades, storage climbs | Retention/decay policy from day one (§6.1) |
| Training job starves a serving replica | Latency breach | Pin GPUs explicitly; schedule training off-peak |
| Prompt injection reaches backend tools | Data exposure | Per-tool scoped tokens; the LLM never holds a superuser credential (§5.2) |
| RL checkpoint regresses production | Quality incident | Automated eval gate with p99 latency check; no direct RL→prod path (§7.3) |

---

## 12. Recommendations

1. **Procure one `p5.48xlarge` on a 3-year Reserved commitment** — $208,286/year, versus $482,150 on-demand. Allocate all 8 GPUs: 2 live, 1 standby, 1 canary, 4 training.
2. **Price `p5e` (H200) against `p5` before committing** (§10.2). ~15% more for meaningfully better SLA margin.
3. **Serve at FP8, TP=1.** Benchmark INT4 separately as a cost tier.
4. **Implement all three guards** — iteration cap, token budget, and wall-clock deadline. The deadline is the only hard SLA guarantee.
5. **Stream over SSE** and **dispatch independent tools in parallel**. Both cut perceived latency and GPU-seconds.
6. **Use least-outstanding-requests with prefix-hash affinity** at the inference LB. Not round-robin.
7. **Train with LoRA**, on the spare GPUs, off-peak. Full-parameter RL is not viable on one GPU and does not need to be.
8. **Gate every promotion** on an automated eval suite including p99 latency.
9. **Set a knowledge-graph retention policy before go-live**, not after.
10. **Make the build-vs-buy decision explicitly** (§9) before capital is committed.

---

## 13. Validate before sign-off

- [ ] Benchmark Qwen3.6-35B-A3B on a single H100 at FP8 — confirm ~200 tok/s single-stream decode and ~96 KV slots at 8K context
- [ ] **Measure p99 end-to-end latency under peak concurrency against the 8 s ceiling** — this is the acceptance criterion, not throughput
- [ ] Measure the real prefix-cache hit rate under agentic traffic (assumed ~50% prefill saving)
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
| 5 | Per-stream decode at batch 12 | ~180 tok/s | Directly sets end-to-end latency |
| 6 | Model config | Qwen3-30B-A3B-class (48 layers, 4 KV heads, head_dim 128) | Changes KV/token and slot count |
| 7 | Decode wall-time per conversation | ~5 s | Sets concurrent stream count via Little's Law |
| 8 | Knowledge-graph extraction density | ~10 entities + 20 edges per conversation | Linear on graph growth and storage |
| 9 | Training MFU | 40% | Linear on continued-pre-training throughput |
| 10 | Scheduled ML model runtime | ~4 h/day | Linear on that line item only (~$159/mo) |
| 11 | Commercial API comparison rate | $0.50/M in, $1.50/M out | Shifts the build-vs-buy multiple in §9 |
| 12 | Utilisation basis | 730 hours/month | Standard AWS month |

**All figures are back-of-the-envelope estimates.** They are directionally correct and sufficient to scope a purchase order, but the GPU count and the latency SLA must be confirmed by load-testing the actual model on the actual hardware before procurement.

---

## Sources

Pricing verified August 2026, us-east-1:

- [p5.48xlarge pricing and specs — Vantage](https://instances.vantage.sh/aws/ec2/p5.48xlarge) — $55.040/hr on-demand, $23.777/hr 3-year reserved
- [p5.48xlarge Specs & Pricing — DoiT Compute](https://www.doit.com/compute/spot/us-east-1/p5.48xlarge)
- [p6-b200.48xlarge Specs & Pricing — DoiT Compute](https://www.doit.com/compute/spot/us-east-1/p6-b200.48xlarge) — $113.9328/hr on-demand
- [p5en.48xlarge pricing and specs — CloudPrice](https://cloudprice.net/aws/ec2/instances/p5en.48xlarge) — 8× H200 141 GB, ~$63.296/hr
- [p5e.48xlarge pricing and specs — Vantage](https://instances.vantage.sh/aws/ec2/p5e.48xlarge) — listed on-demand figure appears inconsistent with its own spot rate; **confirm p5e pricing directly with AWS**
- [AWS quietly increases prices for H200 EC2 instances by 15% — Data Center Dynamics](https://www.datacenterdynamics.com/en/news/aws-quietly-increases-prices-for-h200-ec2-instances-by-15/) — H200 rates have moved recently; re-verify at purchase
- [Prerequisites for Capacity Blocks — AWS documentation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-prerequisites.html)

Non-GPU instance and managed-service rates in §8.2 and §8.3 are list-price estimates and are **not** individually verified. Confirm them in the AWS Pricing Calculator for the target region before issuing a purchase order.
