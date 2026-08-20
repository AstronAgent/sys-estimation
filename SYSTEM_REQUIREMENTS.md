# System Requirements — Qwen3.6-35B-A3B LLM Platform on AWS

**Prepared for:** Client
**Version:** v4 — supersedes v3. Adds the single-GPU RL / pre-training assessment (H200 or B200).
**Date:** 2026-08-20
**Companions:**
- `PLATFORM_REPORT.md` — client-facing report with the full AWS cost breakdown
- `deployment-architecture.md` — diagrams as Mermaid (renders on GitHub/GitLab)
- `deployment-architecture.html` — same diagrams as inline SVG, with PNG/PDF export

> **Headline change from v2.** The model is **Qwen3.6-35B-A3B — a Mixture-of-Experts model: 35B total parameters, ~3B active per token.** That is not a naming detail; it changes the hardware bill materially. VRAM is still sized by *total* parameters, but compute and memory bandwidth are sized by *active* parameters. Decode runs roughly 4–5× faster than a dense 35B, and each replica now fits on **one** H100 instead of two.
>
> **Live fleet drops from 6× H100 to 3× H100.**

---

## 1. Client answers incorporated

| # | Question | Client answer | Effect on design |
|---|---|---|---|
| 1 | 5K tokens per conversation or per turn? | **Per conversation** | v2 assumption confirmed. Decode volume stands at 30M tok/day. |
| 2 | Dedicated ML model — inline or training-only? | **Runs on a fixed frequency; results cached; LLM reads from cache** | Removed from the request path entirely. Becomes a scheduled job + a cache read. Big latency win. |
| 3 | Graph DB scope? | **Full knowledge graph, continuous ingestion, +8 vCPU for weight calculation** | Now sizeable. Separate ingestion worker added. Growth policy required — see §6.3. |
| 4 | Target response time? | **2–8 seconds** | **This is now the binding constraint on replica count, not throughput.** See §4. |
| 5 | Pre-training spec? | **The 16 GB/64 GB/16 vCPU spec belongs to the ML model, not pre-training** | Clarified; RL and pre-training hardware answered below. |
| 6 | Cloud? | **AWS** | Full instance mapping in §7. |
| 7 | Model? | **Qwen/Qwen3.6-35B-A3B** | MoE. Rewrites §3. |
| 8 | RL hardware? | **1× H200 or 1× B200** | **Viable for LoRA RL. Full-parameter does not fit — see §8.1.** |
| 9 | Pre-training hardware? | **1× H200 or 1× B200** | **Viable for LoRA continued pre-training. From-scratch is not feasible — see §8.2.** |

---

## 2. Load model (unchanged from v2, now confirmed)

| Input | Value |
|---|---|
| Daily active users | 3,000 |
| Conversations per user per day | 10 → **30,000 conversations/day** |
| Tokens per conversation | 5,000 (**confirmed: per conversation**) |
| LLM passes per conversation | 2–3 typical, **6 hard cap** |
| Peak concurrent sessions | **~750** (revised down from 2,000) |
| Peak factor | 3.5× |

**Agentic amplification.** Each loop iteration re-sends the accumulated context, so prefill grows quadratically with iteration count while decode grows linearly:

| Pass | Input (cumulative) | Output |
|---|---|---|
| 1 — user turn → tool call | 1,800 | 150 |
| 2 — + tool result → tool call | 3,400 | 150 |
| 3 — + tool result → answer | 5,000 | 700 |
| **Prefill, uncached** | **10,200** | |
| **Prefill, with RadixAttention prefix cache** | **~5,000** | |
| **Decode** | | **1,000** |

| Metric | Per day | Avg | Peak (3.5×) |
|---|---|---|---|
| Conversations | 30,000 | 0.35/s | **1.2/s** |
| LLM inference calls | 75,000 | 0.87/s | ~3/s |
| **Decode tokens** | **30 M** | 350 tok/s | **~1,200 tok/s** |
| Prefill, cached | 150 M | 1,750 tok/s | ~6,100 tok/s |

**Concurrency.** By Little's Law, concurrent *decoding* streams = 1.2/s arrival × ~5 s decode ≈ **~7–25 streams** depending on batch pressure. Of 750 open sessions, roughly **3% are generating at any instant**.

> **Revising concurrent sessions from 2,000 to 750 does not change GPU sizing.** Little's Law runs off the *arrival rate* (30,000 conversations/day × 3.5× peak = 1.2/s), not the number of open sessions. Sessions are held in Redis and cost almost nothing; only actively decoding streams consume HBM. The change reduces the Redis working set and the Chat API connection count — nothing on the GPU tier.

> **750 is also the more internally consistent figure.** At 1.2 conversations/s peak arrival, 750 concurrent sessions implies an average session length of ~10.4 minutes — plausible for a chat with 2–6 agentic turns. The earlier 2,000 figure would have implied ~28-minute sessions.

---

## 3. GPU sizing — MoE changes the answer

### 3.1 What A3B means for hardware

| Dimension | Governed by | Value |
|---|---|---|
| VRAM for weights | **Total** params (35B) | ~35 GB at FP8, ~70 GB at BF16 |
| Compute (FLOPs/token) | **Active** params (3B) | ~11× less than dense 35B |
| Memory bandwidth per token | **Active** params (3B) | ~3 GB read/token at FP8 |
| KV cache | Layers × KV heads | ~48 KB/token at FP8 (see 3.2) |

Decode on a modern LLM is memory-bandwidth-bound. Reading 3 GB/token instead of 35 GB/token is why this model decodes ~4–5× faster than a dense 35B on the same GPU.

### 3.2 Fits on one H100 — precision is the deciding factor

| Precision | Weights | KV headroom on 1× H100 80 GB | Verdict |
|---|---|---|---|
| BF16 | ~70 GB | ~2 GB — unusable | Would force TP=2 purely for memory |
| **FP8** | **~35 GB** | **~37 GB → ~96 streams @ 8K ctx** | **Recommended. TP=1, one GPU per replica.** |
| INT4 | ~18 GB | ~54 GB | Cost tier; benchmark quality first |

KV cache per token (assuming a Qwen3-30B-A3B-class config — 48 layers, 4 KV heads, head_dim 128; **confirm against the actual model card**):

```
2 (K+V) × 48 layers × 4 kv_heads × 128 head_dim × 1 byte (FP8) = 49,152 B = 48 KiB/token
→ 8K context ≈ 384 MB per active stream
→ (80 − 35 weights − 8 overhead) = 37 GB ÷ 0.384 GB ≈ 96 concurrent streams per replica
```

**Consequence:** TP=1 at FP8. The NVLink-within-one-chassis constraint from v2 **no longer applies per replica** — each replica is a single GPU. It still applies if you choose BF16.

### 3.3 Throughput per replica (1× H100, FP8, TP=1)

| Metric | Estimate |
|---|---|
| Single-stream decode | ~200–250 tok/s |
| Per-stream decode at batch ~12 | ~150–180 tok/s |
| Per-stream decode at batch ~25 | ~110–130 tok/s |
| Aggregate decode | ~2,000–4,000 tok/s |
| Prefill | ~20,000–30,000 tok/s |

Peak decode demand is ~1,200 tok/s. **One replica clears it on throughput alone.** Throughput is no longer what sets the replica count — latency is.

---

## 4. Latency budget — the binding constraint (2–8 s SLA)

With the ML model moved off the request path (client answer 2), a 3-pass conversation looks like this:

| Step | Time |
|---|---|
| Pass 1 prefill (1,800 tok @ ~25k tok/s) | 0.07 s |
| Pass 1 decode (150 tok @ 180 tok/s) | 0.83 s |
| Tool round 1 (vector / graph / cache read) | 0.15 s |
| Pass 2 prefill (+1,600, prefix cached) | 0.06 s |
| Pass 2 decode (150 tok) | 0.83 s |
| Tool round 2 | 0.15 s |
| Pass 3 prefill (+1,600) | 0.06 s |
| **Pass 3 decode (700 tok — the final answer)** | **3.9 s** |
| Orchestrator + network overhead | 0.30 s |
| **End-to-end** | **~6.4 s** |

Inside the 2–8 s window, with ~1.6 s of slack. Two findings follow.

### 4.1 The 6-iteration cap breaches the SLA

Three additional tool rounds add ~3.2 s → **~9.6 s end-to-end**. A conversation that hits the iteration cap will miss the 8-second ceiling. Four mitigations, all recommended:

1. **Stream the final answer (SSE).** Time-to-first-token is ~2.6 s; the user sees output flowing well inside the window even when total generation runs longer. This is the single highest-leverage fix.
2. **Wall-clock deadline in the orchestrator.** At 6 s elapsed, stop issuing tool calls and force a final answer from what has been gathered. This is a hard guarantee against the ceiling; the iteration cap alone is not.
3. **Parallel tool calls.** Independent tools issued in one round instead of two saves a full pass (~1 s each). Model this in the tool graph — it is exactly what the graph edges are for.
4. **Keep batch sizes low** so per-stream decode stays fast.

### 4.2 Latency sets the replica count

| Config | Batch/replica at peak | Per-stream decode | Final-answer time | End-to-end | SLA |
|---|---|---|---|---|---|
| 1 active replica | ~25 | ~120 tok/s | 5.8 s | **~8.5 s** | ✗ breaches |
| **2 active replicas** | ~12 | ~180 tok/s | 3.9 s | **~6.4 s** | ✓ meets |
| 3 active replicas | ~8 | ~200 tok/s | 3.5 s | ~6.0 s | ✓ marginal gain |

**Recommendation: 2 active + 1 standby = 3× H100 80 GB, TP=1.**

The standby covers N+1 failover and gives hot-swap a staging target without a capacity dip. Note this is set by the latency SLA — on raw throughput, one replica would do.

---

## 5. AWS deployment

### 5.1 GPU instances

AWS sells H100 in the **p5** family. `p5.48xlarge` (8× H100 80 GB, 192 vCPU, 2 TB RAM) is the standard SKU; confirm whether smaller p5 variants are available in your target region, and price against **EC2 Capacity Blocks for ML** and 1-year Savings Plans — on-demand p5 is the most expensive way to buy this.

| Role | Instance | GPU allocation |
|---|---|---|
| **Live LLM fleet** | 1× `p5.48xlarge` | 2 active + 1 standby replica = **3 of 8 GPUs** |
| Production Testing (canary) | Same instance, **1 GPU** | 4 of 8 GPUs still free for burst and swap staging |
| Dedicated ML model (scheduled) | `g6.4xlarge` (1× L4 24 GB, 16 vCPU, 64 GB) | Exact match to the client spec. Scheduled, not always-on. |
| RL post-training (LoRA) | Spare GPU on the same `p5.48xlarge`, off-peak | 1 GPU — see §8.1 |

**One `p5.48xlarge` carries the entire live platform plus canary, with 4 GPUs spare.** That is one instance, not a cluster.

*Isolation note:* co-locating the canary with production on one instance is the cost-efficient choice but weakens blast-radius isolation. If the client requires hard separation, move Production Testing to a second smaller instance — but be aware that a lower-bandwidth GPU (L40S, L4) will not reproduce production latency, so performance testing must still run on H100.

### 5.2 Everything else

| Component | AWS service / instance | Sizing basis |
|---|---|---|
| Edge load balancer | ALB + AWS WAF | TLS, rate limiting. ~3 req/s peak — trivial. |
| Auth | Cognito, or Keycloak on 2× `m7g.large` | OIDC/OAuth2, JWT + refresh |
| Chat API | ECS/EKS, 2–4× `m7g.xlarge` | Stateless, SSE streaming, autoscaled |
| MCP Graph Orchestrator | 3× `m7g.2xlarge` (8 vCPU / 32 GB) | Holds the agentic loop; CPU-bound |
| Async task queue | SQS | Decouples long agentic runs from the HTTP tier |
| Session cache | ElastiCache Redis, `cache.r7g.large` + replica | 750 sessions × ~200 KB ≈ **150 MB** |
| Chat history / auth | RDS PostgreSQL `db.r7g.2xlarge`, Multi-AZ, 1 TB gp3 | ~600 MB/day → **~220 GB/yr**. ~5 QPS. No sharding. |
| Vector DB | OpenSearch Serverless (vector engine), or Qdrant on 3× `r7g.xlarge` | ~600 MB/day → **~500 GB/yr with index overhead** |
| **Graph DB** | Neptune `db.r6g.2xlarge`, or Neo4j on `r7g.2xlarge` + replica | See §6.3 — needs a growth policy |
| **Graph ingestion worker** | `c7g.2xlarge` (**8 vCPU** / 16 GB) | Per client answer 3 — edge-weight calculation |
| ML result cache | DynamoDB with native TTL (+ Redis if read QPS demands) | Per client answer 2 |
| Checkpoints / artifacts | S3, ~5 TB | 35 GB per FP8 checkpoint; 50 retained ≈ 2 TB |
| Observability | CloudWatch + AMP + AMG, OTel traces | Per-token latency, KV utilisation, cache hit rate, iteration histogram |

---

## 6. Subsystem detail

### 6.1 MCP harness

- MCP servers are **graph nodes**; **edges define allowed call transitions**. Route tool calls through the graph rather than exposing a flat tool list.
- **Model independent tools as parallel-dispatchable edges** — per §4.1, this is a latency feature, not just a modelling nicety.
- **Two auth layers.** User-facing: OIDC/OAuth2 + JWT with refresh, per-user rate limits. Service-to-service: **per-tool scoped tokens**. The LLM must never hold one credential that reaches every backend — a prompt-injected tool call would otherwise reach the whole data tier.
- **Three cost/latency guards, all required:** iteration cap (6), per-conversation token budget, and wall-clock deadline (6 s).

### 6.2 Dedicated ML model — scheduled, cached (client answer 2)

The model runs on a fixed frequency, writes its inference output to a cache, and the LLM reads that cache through an MCP tool. It is **not** on the request path.

- **Compute:** `g6.4xlarge` — 1× L4 24 GB, 16 vCPU, 64 GB RAM. Matches the stated spec. Trigger on EventBridge schedule; stop the instance between runs, or run on Spot.
- **Cache:** DynamoDB with native TTL. Read latency single-digit ms, so the MCP tool call costs effectively nothing against the latency budget.
- **Required policy decisions (new, see §8):** what frequency, what maximum acceptable staleness, and what happens on a cache miss — serve stale, trigger an on-demand run, or return "unavailable" and let the LLM proceed without it. The third option is usually correct for a latency-bound system.

### 6.3 Knowledge graph — continuous ingestion (client answer 3)

Full knowledge graph, continuously ingested, with a dedicated 8 vCPU worker for weight calculation.

Estimated growth (assuming ~10 entities and ~20 edges extracted per conversation — **confirm this extraction density**):

| Metric | Per day | Per year |
|---|---|---|
| Nodes | 300,000 | ~110 M |
| Edges | 600,000 | ~220 M |
| Storage (with indexes) | ~120 MB | **~45 GB** |
| Ingestion write rate | 3.5 nodes/s avg | ~12/s peak |

Storage and write rate are both comfortable. **The concern is the weight calculation.**

> **Recommendation — incremental, not full-graph, weight computation.** 8 vCPU is ample for incremental or windowed edge-weight updates on a stream of 12 writes/second. It is **not** sufficient to recompute a global metric (PageRank, centrality, community detection) over a 110-million-node graph — that is a distributed-compute job, and at year-two scale it would run for hours on 8 cores. If global weights are genuinely required, either schedule them as periodic batch jobs on separate compute (Neptune Analytics, or Spark on EMR), or approximate them incrementally.
>
> **A continuously ingesting graph with no retention policy grows without bound.** Add decay, pruning, or entity-level consolidation from day one; retrofitting it onto a 200M-edge graph is painful.

---

## 7. Training and hot-swap

```
Pre-training → Post-training (RL) → S3 Model Registry
    → Production Testing (canary + shadow traffic)
    → Automated Eval Gate
    → promote → hot-swap into live SGLang fleet
                     ↳ production traces + reward signal feed back into RL
```

**Hot-swap.** Preferred: SGLang's live weight-update path (`update_weights_from_disk` / distributed weight sync — the same mechanism used inside RLHF loops). Weights are replaced in-place in a running server; connections stay open, no cold start. Fallback: blue/green drain through the inference load balancer, with automatic rollback on regression. The standby replica exists partly so this happens without a capacity dip.

**Promotion gate — non-negotiable.** Every checkpoint clears an automated eval suite (safety, regression, task accuracy, **p99 latency against the 8 s SLA**, tool-call validity) before reaching live traffic. There must be **no direct path from the RL loop into the live environment**. A continuous training pipeline that can push straight to production is a continuous outage pipeline.

---

## 8. Training hardware — single-GPU assessment

Client specifies **1× H200 or 1× B200** for both RL post-training and pre-training. Short answer: **viable with LoRA, not viable full-parameter, and not viable for pre-training from scratch.** The numbers follow.

| | H200 SXM | B200 SXM |
|---|---|---|
| VRAM | 141 GB HBM3e | 180 GB HBM3e |
| Memory bandwidth | 4.8 TB/s | 8.0 TB/s |
| BF16 dense (peak) | ~989 TFLOPS | ~2,250 TFLOPS |
| FP8 (peak) | ~1,979 TFLOPS | ~4,500 TFLOPS |

### 8.1 RL post-training on one GPU

Memory budget for RL on Qwen3.6-35B-A3B (GRPO/RLOO-style; PPO adds a critic and is worse):

| Approach | Policy | Reference | Grads | Optimizer | Activations + rollouts | **Total** | H200 141 GB | B200 180 GB |
|---|---|---|---|---|---|---|---|---|
| Full-param BF16 + AdamW FP32 | 70 | 70 | 70 | 280 | ~30 | **~520 GB** | ✗ | ✗ |
| Full-param FP8 + 8-bit Adam | 35 | 35 | 35 | 70 | ~30 | **~205 GB** | ✗ | ✗ |
| **LoRA on FP8 base** | 35 | **0** | ~1 | ~4 | ~25 | **~66 GB** | **✓** | **✓** |
| QLoRA (NF4 base) | 18 | **0** | ~1 | ~4 | ~25 | **~48 GB** | ✓ | ✓ |

Two things drive that table:

- **AdamW FP32 optimizer state is 8 bytes per trainable parameter** — 280 GB on 35B params. It is the largest single term, and it is why full-parameter RL cannot fit on any single GPU that exists today.
- **With LoRA the reference model is free.** The KL-penalty reference is the base model with adapters disabled — same weights, no second copy. Full-parameter RL needs an actual second model resident in memory.

**Verdict: LoRA RL fits comfortably on one H200 or B200. Full-parameter RL does not, and no amount of quantisation gets it there on a single device.**

**MoE-specific guidance:** apply LoRA to the attention projections (q, k, v, o). Adapting all expert FFNs is expensive and rarely necessary for alignment work. The router can also be adapted if expert-selection behaviour must change — but router training destabilises easily, so start without it.

**Throughput — RL is rollout-generation-bound, not gradient-bound:**

| | Single-stream decode | Batched rollout | GRPO step (16 rollouts × 1k tok) | Steps/day |
|---|---|---|---|---|
| H200 | ~300 tok/s | ~4,000 tok/s | ~4 s gen + ~2 s fwd/bwd | **~600–1,200** |
| B200 | ~500 tok/s | ~7,000 tok/s | ~2.5 s gen + ~1 s fwd/bwd | **~1,000–2,000** |

Sufficient for **continuous incremental RL from production traces**, which is what the pipeline describes. Not sufficient for a from-scratch alignment campaign on a large preference dataset — budget weeks for that, not days.

### 8.2 Pre-training on one GPU

| Scenario | Memory needed | Time on 1 GPU | Verdict |
|---|---|---|---|
| **From scratch**, 35B MoE, ~2T tokens | ~560 GB | **~2.9 yr (H200) / ~1.3 yr (B200)** | **Not feasible** |
| Full-parameter continued pre-training | ~560 GB | — | **Not feasible on memory** |
| **LoRA continued / domain-adaptive PT** | ~66 GB | see below | **Viable** |

*From-scratch compute derivation:* training FLOPs ≈ 6 × active_params × tokens = 6 × 3e9 × 2e12 = **3.6e22 FLOPs**. At 40% MFU that is ~400 TFLOPS effective on H200 and ~900 on B200 → 9.0e7 s and 4.0e7 s respectively. Memory fails first regardless: weights 70 + grads 70 + FP32 master 140 + AdamW 280 ≈ **560 GB**.

*LoRA continued-pre-training throughput:*

```
FLOPs/token ≈ 6 × 3e9 active params = 1.8e10
H200 @ ~400 TFLOPS effective → ~22,000 tok/s → ~1.9 B tokens/day
B200 @ ~900 TFLOPS effective → ~50,000 tok/s → ~4.3 B tokens/day
```

**A 10–50 B token domain corpus takes roughly 5–26 days on H200, or 2–12 days on B200.** That is a genuinely useful capability and almost certainly what is intended — domain adaptation, not building a foundation model.

**Confirm that "pre-training" means domain-adaptive continued pre-training.** If it means training a base model from scratch, the requirement needs hundreds of GPUs and a fundamentally different budget.

### 8.3 You cannot buy one H200 or one B200 on AWS

Both ship only as eight-GPU instances:

| GPU | AWS instance | GPUs |
|---|---|---|
| H200 | `p5e.48xlarge` | 8 |
| B200 | `p6-b200.48xlarge` | 8 |

"1× H200" is therefore not a purchasable unit — it is one GPU of an eight-GPU machine.

> **Recommendation: do not buy a separate training instance.** The `p5.48xlarge` already in the plan has **4 spare H100s**. LoRA RL needs ~66 GB, which fits on **one H100 80 GB** with headroom. Run RL and continued pre-training on the spare GPUs, scheduled off-peak. This removes an entire instance from the bill.
>
> The trade-off is isolation: training and serving share a machine. Because serving peaks in business hours and RL is a background process, off-peak scheduling resolves most of the contention — but pin GPUs explicitly so a training job cannot starve a serving replica and breach the 8-second SLA.
>
> If hard isolation from production is required, buy separately and accept the cost.

### 8.4 Worth pricing: H200 for *serving*, not only training

If the serving fleet moved from `p5.48xlarge` (H100) to `p5e.48xlarge` (H200):

| | H100 80 GB | H200 141 GB |
|---|---|---|
| KV headroom after 35 GB weights | ~37 GB → **~96 slots** | ~98 GB → **~255 slots** |
| Decode speed (bandwidth-proportional) | 1.0× | **~1.43×** |
| Per-stream decode at batch 12 | ~180 tok/s | **~260 tok/s** |
| Modelled end-to-end response | ~6.4 s | **~5.2 s** |

More margin against the 8-second ceiling, and it may allow dropping to **1 active + 1 standby**. Since the same instance would also host RL and continued pre-training with far more headroom, `p5e` is worth pricing directly against `p5` before committing.

### 8.5 Still needed to finalise sizing

1. **Exact Qwen3.6-35B-A3B config** — layer count, KV heads, head_dim, expert count. §3.2's KV math assumes a Qwen3-30B-A3B-class shape.
2. **ML model frequency and staleness tolerance** — run cadence, maximum acceptable result age, and cache-miss behaviour (§6.2).
3. **Knowledge-graph extraction density** — entities and edges per conversation, which drives §6.3's growth numbers.
4. **Whether graph weight calculation is incremental or global** — 8 vCPU supports the former, not the latter at scale (§6.3).
5. **Confirmation that "pre-training" means continued/domain-adaptive** (§8.2).
6. **Region and reserved-capacity strategy** — p5 / p5e / p6 availability varies by region and materially affects cost and lead time.

---

## 9. Bill of materials

| Item | AWS | Qty |
|---|---|---|
| Live LLM fleet + canary | `p5.48xlarge` (8× H100 80 GB) | **1** (3 GPUs live, 1 canary, 4 spare) |
| Dedicated ML model (scheduled) | `g6.4xlarge` | 1 |
| RL post-training (LoRA) | **Spare H100 on the existing `p5.48xlarge`** — or `p5e.48xlarge` (H200) if isolation is required | 1 GPU |
| Continued pre-training (LoRA) | Same spare-GPU pool, scheduled off-peak | 1 GPU |
| Pre-training from scratch | **Not feasible on 1 GPU — see §8.2** | — |
| MCP orchestrator | `m7g.2xlarge` | 3 |
| Chat API | `m7g.xlarge` | 2–4 (autoscaled) |
| Auth | Cognito or `m7g.large` | 2 |
| Graph ingestion worker | `c7g.2xlarge` (8 vCPU) | 1 |
| PostgreSQL | RDS `db.r7g.2xlarge` Multi-AZ | 1 + replica |
| Redis | ElastiCache `cache.r7g.large` | 1 + replica |
| Vector DB | OpenSearch Serverless or 3× `r7g.xlarge` | 3 |
| Graph DB | Neptune `db.r6g.2xlarge` or Neo4j on `r7g.2xlarge` | 1 + replica |
| ML result cache | DynamoDB (TTL) | — |
| Queue | SQS | — |
| Object storage | S3 | ~5 TB |
| Load balancing | ALB + WAF | — |

---

## 10. Validate before sign-off

- [ ] Benchmark **Qwen3.6-35B-A3B on a single H100 at FP8** — confirm ~200 tok/s single-stream decode and ~96 KV slots at 8K context
- [ ] **Measure p99 end-to-end latency under peak concurrency against the 8 s ceiling** — this is the acceptance criterion, not throughput
- [ ] Measure the real prefix-cache hit rate under agentic traffic (assumed ~50% prefill saving)
- [ ] Measure the actual distribution of iterations per conversation (assumed avg 2.5, cap 6) — if the tail is fatter than modelled, §4.1's deadline mechanism carries the SLA
- [ ] Confirm SSE streaming delivers TTFT under ~3 s
- [ ] Load-test knowledge-graph ingestion at 12 writes/s with weight calculation running concurrently on 8 vCPU
- [ ] Rehearse a hot-swap under live traffic — confirm zero dropped connections
- [ ] Rehearse an eval-gate failure — confirm automatic rollback
