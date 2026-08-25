# System Requirements — Qwen3.6-35B-A3B LLM Platform on AWS

**Prepared for:** Client
**Version:** v5 — supersedes v4. Serving hardware moved from H100 (`p5.48xlarge`) to **H200 (`p5e.48xlarge`)**. §3, §4, §5.1, §8 and §9 all change.
**Date:** 2026-08-20
**Companions:**
- `PLATFORM_REPORT.md` — client-facing report with the full AWS cost breakdown
- `deployment-architecture.md` — diagrams as Mermaid (renders on GitHub/GitLab)
- `deployment-architecture.html` — same diagrams as inline SVG, with PNG/PDF export

> **Headline change from v4 — H200 replaces H100.** The serving fleet moves from `p5.48xlarge` (8× H100 80 GB) to **`p5e.48xlarge` (8× H200 141 GB)**. H200 is the same GH100 die with a different memory subsystem: 141 GB of HBM3e at **4.8 TB/s** instead of 80 GB of HBM3 at 3.35 TB/s. Decode is memory-bandwidth-bound, so it runs **~1.43× faster**; prefill is compute-bound and is unchanged.
>
> **What that buys:** ~4.6 s modelled end-to-end instead of ~6.4 s, the 6-iteration worst case falls from ~9.6 s (a breach) to ~7.0 s (inside the ceiling), single-replica operation becomes viable at ~6.7 s, KV capacity per replica goes from ~96 to ~255 slots, and the client's stated "1× H200" training requirement is met by a spare GPU on the serving box rather than by substituting an 80 GB card.
>
> **What it costs:** ~$31,247/year more on a 3-year reserved commitment — about 15%. See §8.4 and `PLATFORM_REPORT.md` §10.2.

> **Headline change from v2, retained.** The model is **Qwen3.6-35B-A3B — a Mixture-of-Experts model: 35B total parameters, ~3B active per token.** That is not a naming detail; it changes the hardware bill materially. VRAM is still sized by *total* parameters, but compute and memory bandwidth are sized by *active* parameters. Decode runs roughly 4–5× faster than a dense 35B, and each replica fits on **one** GPU instead of two.
>
> **Live fleet drops from 6 GPUs to 3.**

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
| 8 | RL hardware? | **1× H200 or 1× B200** | **Viable for LoRA RL. Full-parameter does not fit — see §8.1.** Now satisfied exactly: the serving instance is 8× H200 with 4 spare. |
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

### 3.2 Fits on one H200 — and precision is now a speed decision, not a memory one

| Precision | Weights | KV headroom on 1× H200 141 GB | Verdict |
|---|---|---|---|
| BF16 | ~70 GB | ~63 GB → ~164 streams | Fits at TP=1 on 141 GB, where it was impossible on 80 GB. Still rejected: 2× the bytes read per token halves decode → ~130 tok/s at batch 12 → ~8.5 s end-to-end. |
| **FP8** | **~35 GB** | **~98 GB → ~255 streams @ 8K ctx** | **Recommended. TP=1, one GPU per replica.** |
| INT4 | ~18 GB | ~115 GB → ~300 streams | Cost tier; benchmark quality first |

KV cache per token (assuming a Qwen3-30B-A3B-class config — 48 layers, 4 KV heads, head_dim 128; **confirm against the actual model card**):

```
2 (K+V) × 48 layers × 4 kv_heads × 128 head_dim × 1 byte (FP8) = 49,152 B = 48 KiB/token
→ 8K context ≈ 384 MB per active stream
→ (141 − 35 weights − 8 overhead) = 98 GB ÷ 0.384 GB ≈ 255 concurrent streams per replica
```

**Consequence:** TP=1 at FP8. The NVLink-within-one-chassis constraint from v2 **does not apply per replica** — each replica is a single GPU. On H200 it would not apply at BF16 either, since BF16 now fits on one card; BF16 is excluded on bandwidth instead.

> **The capacity gain is not why H200 was chosen.** 96 slots per replica already covered the ~25 concurrent streams of peak demand by ~4×. The reason is the 4.8 TB/s, which is what §4 spends. The 255 slots matter later — for longer contexts, for higher pass counts, and for co-resident training — not for the load as specified.

### 3.3 Throughput per replica (1× H200, FP8, TP=1)

| Metric | H100 80 GB | **H200 141 GB** | Why it scales that way |
|---|---|---|---|
| Single-stream decode | ~200–250 tok/s | **~290–360 tok/s** | bandwidth-bound → ×1.43 |
| Per-stream decode at batch ~12 | ~150–180 tok/s | **~215–260 tok/s** | bandwidth-bound → ×1.43 |
| Per-stream decode at batch ~25 | ~110–130 tok/s | **~160–185 tok/s** | bandwidth-bound → ×1.43 |
| Aggregate decode | ~2,000–4,000 tok/s | **~2,900–5,700 tok/s** | bandwidth-bound → ×1.43 |
| Prefill | ~20,000–30,000 tok/s | **~20,000–30,000 tok/s** | compute-bound → **no change** |

> **Prefill deliberately does not improve.** H200 and H100 SXM carry the same GH100 die — same SM count, same ~1,979 TFLOPS FP8 peak. Only HBM changed. Decode reads ~3 GB of weights per token and is limited by bandwidth, so it takes the full 4.8 ÷ 3.35 = **1.43×**. Prefill is arithmetic-bound and takes none of it. Every latency gain in §4 comes from decode terms only — if a benchmark shows prefill speeding up too, the model is not running the configuration assumed here.

Peak decode demand is ~1,200 tok/s. **One replica clears it on throughput alone, with roughly 3–5× to spare.** Throughput is not what sets the replica count — latency is.

---

## 4. Latency budget — the binding constraint (2–8 s SLA)

With the ML model moved off the request path (client answer 2), a 3-pass conversation looks like this:

| Step | H100 | **H200** |
|---|---|---|
| Pass 1 prefill (1,800 tok @ ~25k tok/s) | 0.07 s | 0.07 s |
| Pass 1 decode (150 tok @ 180 → 260 tok/s) | 0.83 s | **0.58 s** |
| Tool round 1 (vector / graph / cache read) | 0.15 s | 0.15 s |
| Pass 2 prefill (+1,600, prefix cached) | 0.06 s | 0.06 s |
| Pass 2 decode (150 tok) | 0.83 s | **0.58 s** |
| Tool round 2 | 0.15 s | 0.15 s |
| Pass 3 prefill (+1,600) | 0.06 s | 0.06 s |
| **Pass 3 decode (700 tok — the final answer)** | 3.9 s | **2.69 s** |
| Orchestrator + network overhead | 0.30 s | 0.30 s |
| **End-to-end** | ~6.4 s | **~4.6 s** |

Inside the 2–8 s window with **~3.4 s of slack**, up from ~1.6 s on H100. Three findings follow.

### 4.1 The 6-iteration cap no longer breaches the SLA — but the margin is thin

Three additional tool rounds add ~0.79 s each on H200 (0.15 tool + 0.06 prefill + 0.58 decode), against ~1.04 s each on H100:

| | H100 | **H200** |
|---|---|---|
| 3-pass typical | ~6.4 s | **~4.6 s** |
| 6-pass worst case | **~9.6 s — breaches** | **~7.0 s — inside** |
| Margin at worst case | −1.6 s | **+1.0 s** |

**This is the single biggest behavioural change from the H100 plan.** On H100 the worst case was an outright SLA violation and the wall-clock deadline was load-bearing — it was the only thing preventing a breach. On H200 the worst case fits unaided, and the deadline becomes a safety net.

**Keep all four mitigations anyway.** One second of margin will be consumed by a slow graph query, a prefix-cache miss, or a GC pause:

1. **Stream the final answer (SSE).** Time-to-first-token is ~1.9 s on H200 (from ~2.6 s); the user sees output flowing well inside the window even when total generation runs longer. This remains the single highest-leverage fix.
2. **Wall-clock deadline in the orchestrator.** At 6 s elapsed, stop issuing tool calls and force a final answer from what has been gathered. This is the only hard guarantee against the ceiling; the iteration cap alone is not, on any hardware.
3. **Parallel tool calls.** Independent tools issued in one round instead of two saves a full pass (~0.8 s each on H200). Model this in the tool graph — it is exactly what the graph edges are for.
4. **Keep batch sizes low** so per-stream decode stays fast.

### 4.2 Latency sets the replica count

| Config | Batch/replica at peak | Per-stream decode | Final-answer time | End-to-end | SLA |
|---|---|---|---|---|---|
| 1 active replica | ~25 | ~170 tok/s | 4.1 s | **~6.7 s** | ✓ meets — thin |
| **2 active replicas** | ~12 | ~260 tok/s | 2.7 s | **~4.6 s** | ✓ **recommended** |
| 3 active replicas | ~8 | ~290 tok/s | 2.4 s | ~4.2 s | ✓ marginal gain |

*Batch sizes are held at the H100 concurrency figures (§2's ~25 streams at peak). Faster decode drains the queue faster, so real concurrency on H200 will be lower and per-stream speed correspondingly higher. The table does not take that credit — it is deliberately pessimistic.*

**Recommendation: 2 active + 1 standby = 3× H200 141 GB, TP=1.** Unchanged in shape from the H100 plan, changed in justification.

On H100, the second active replica was **required**: one replica breached at ~8.5 s. On H200, one replica meets the SLA at ~6.7 s, so the second buys **p99 margin rather than compliance** — and it costs nothing marginal, since all 8 GPUs are on the one instance regardless.

The practical consequence is an operational one:

> **Single-replica serving becomes a supported degraded mode.** The fleet can lose a GPU, or drain one for a weight swap, and still answer inside the 2–8 s window. On H100 any single-replica period was an SLA event, which made maintenance windows and failover both riskier than they should have been.

The standby still covers N+1 failover and gives hot-swap a staging target without a capacity dip.

---

## 5. AWS deployment

### 5.1 GPU instances

AWS sells H200 in the **p5e** family. `p5e.48xlarge` (8× H200 141 GB, 192 vCPU, 2 TB RAM) is the SKU; `p5en.48xlarge` is the same GPUs on an Emerald Rapids host with EFAv3 networking and is the variant with the most reliably published rate. There are no smaller p5e variants — 8 GPUs is the minimum purchasable unit. Price against **EC2 Capacity Blocks for ML** and 1-year Savings Plans; on-demand p5e is the most expensive way to buy this.

> **Check regional availability before the design depends on it.** `p5e` is thinner on the ground than `p5` in several regions, so capacity is a lead-time risk as well as a cost question.

| Role | Instance | GPU allocation |
|---|---|---|
| **Live LLM fleet** | 1× `p5e.48xlarge` | 2 active + 1 standby replica = **3 of 8 GPUs** |
| Production Testing (canary) | Same instance, **1 GPU** | 4 of 8 GPUs still free for burst and swap staging |
| Dedicated ML model (scheduled) | `g6.4xlarge` (1× L4 24 GB, 16 vCPU, 64 GB) | Exact match to the client spec. Scheduled, not always-on. |
| RL post-training (LoRA) | Spare **H200** on the same `p5e.48xlarge`, off-peak | 1 GPU — see §8.1. This is the "1× H200" the requirement asked for. |

**One `p5e.48xlarge` carries the entire live platform plus canary, with 4 GPUs spare.** That is one instance, not a cluster.

*Isolation note:* co-locating the canary with production on one instance is the cost-efficient choice but weakens blast-radius isolation. If the client requires hard separation, move Production Testing to a second smaller instance — but be aware that a lower-bandwidth GPU (L40S, L4) will not reproduce production latency, so performance testing must still run on H200. A canary on an H100 would also mismodel latency now, in the optimistic direction: it would look slower than production, which is at least the safe direction to be wrong in.

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

> **Moving serving to H200 resolves the procurement half of this question outright.** In v4 the recommendation was to run training on a spare *H100* — an 80 GB substitute for the 141 GB card the client specified, which worked for LoRA but was a quiet downgrade. Now the serving instance is 8× H200 with **4 spare H200s**, so the requirement is met literally, on hardware already bought for serving. §8.3 changes accordingly.

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

At ~66 GB on a 141 GB card, LoRA RL uses under half the device. That headroom is what makes co-residency with serving practical: larger rollout batches, longer sequences, or a second concurrent experiment all fit without touching the serving replicas.

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

### 8.3 You cannot buy one H200 or one B200 on AWS — but you no longer need to

Both ship only as eight-GPU instances:

| GPU | AWS instance | GPUs |
|---|---|---|
| H200 | `p5e.48xlarge` | 8 |
| B200 | `p6-b200.48xlarge` | 8 |

"1× H200" is not a purchasable unit — it is one GPU of an eight-GPU machine. In v4 this was an awkward finding, because the serving instance was a p5 and the nearest available spare GPU was an 80 GB H100.

> **Recommendation: do not buy a separate training instance.** The `p5e.48xlarge` in the plan has **4 spare H200s**. LoRA RL needs ~66 GB, which fits on **one H200 141 GB** with better than 2× headroom. Run RL and continued pre-training there, scheduled off-peak. This removes an entire instance from the bill **and** delivers the specified hardware exactly rather than an approximation of it.
>
> The trade-off is isolation: training and serving share a machine. Because serving peaks in business hours and RL is a background process, off-peak scheduling resolves most of the contention — but pin GPUs explicitly so a training job cannot starve a serving replica and breach the 8-second SLA. Note that the SLA now has ~3.4 s of slack rather than ~1.6 s, so an imperfectly isolated training job is less likely to cause a visible breach; that is a reason to relax vigilance about *incident severity*, not about pinning.
>
> If hard isolation from production is required, a second `p5e.48xlarge` roughly doubles the GPU bill to use one of eight GPUs. Price that against the isolation it buys before agreeing to it.

### 8.4 H200 versus H100 for serving — the decision, in full

v4 flagged this as "worth pricing". It has now been priced, and H200 is the recommendation. `p5` (H100) remains a costed fallback rather than a rejected option:

| | H100 80 GB (`p5.48xlarge`) | **H200 141 GB (`p5e.48xlarge`)** |
|---|---|---|
| Memory | 80 GB HBM3 | **141 GB HBM3e** |
| Bandwidth | 3.35 TB/s | **4.8 TB/s** |
| Compute (FP8 peak) | ~1,979 TFLOPS | ~1,979 TFLOPS — **same die** |
| KV headroom after 35 GB weights | ~37 GB → ~96 slots | **~98 GB → ~255 slots** |
| Decode speed | 1.0× | **~1.43×** |
| Prefill speed | 1.0× | 1.0× — no gain |
| Per-stream decode at batch 12 | ~180 tok/s | **~260 tok/s** |
| Modelled end-to-end (2 active) | ~6.4 s | **~4.6 s** |
| Slack against 8 s | ~1.6 s | **~3.4 s** |
| 6-iteration worst case | ~9.6 s — **breaches** | **~7.0 s — inside** |
| 1-replica degraded mode | ~8.5 s — breaches | **~6.7 s — holds** |
| Training GPU vs the stated spec | 80 GB — a substitution | **141 GB — as specified** |
| On-demand | $55.040/hr → $482,150/yr | $63.296/hr → **$554,473/yr** |
| 3-yr reserved | $23.777/hr → $208,286/yr | ~$27.344/hr → **$239,533/yr** *(derived)* |
| **Annual delta (3-yr reserved)** | — | **+$31,247 (+15%)** |

**The argument for paying it.** Three of the four SLA-relevant rows change sign, not just magnitude: the 6-iteration worst case, single-replica operation, and the fallback position if the prefix-cache hit rate underperforms. On H100 each of those was a breach that had to be engineered around; on H200 each is absorbed by the hardware. That is worth more than 1.8 seconds of typical-case latency, which is the part most visible in the table and the least important operationally.

**The argument against.** $31,247/year is real money for a measurable but not decisive latency gain, and the p5e reserved rate is derived rather than quoted (see below). If the 6-iteration tail proves rare in measurement and the wall-clock deadline is properly implemented, `p5` delivers the specified SLA at lower cost.

> **The 3-year p5e rate is the number to verify first.** AWS publishes a verified 3-year reserved rate for `p5.48xlarge` ($23.777/hr = 43.2% of its on-demand rate) but no equivalent open rate for `p5e`. The $27.344/hr above applies that same ratio. **Confirm with AWS before committing** — it carries ~$240k/year, and a materially worse discount would widen the delta and reopen this comparison.

Full cost tables in `PLATFORM_REPORT.md` §8 and §10.2.

### 8.5 Still needed to finalise sizing

1. **Exact Qwen3.6-35B-A3B config** — layer count, KV heads, head_dim, expert count. §3.2's KV math assumes a Qwen3-30B-A3B-class shape.
2. **ML model frequency and staleness tolerance** — run cadence, maximum acceptable result age, and cache-miss behaviour (§6.2).
3. **Knowledge-graph extraction density** — entities and edges per conversation, which drives §6.3's growth numbers.
4. **Whether graph weight calculation is incremental or global** — 8 vCPU supports the former, not the latter at scale (§6.3).
5. **Confirmation that "pre-training" means continued/domain-adaptive** (§8.2).
6. **Region and reserved-capacity strategy** — p5e / p5 / p6 availability varies by region and materially affects cost and lead time. `p5e` is the scarcer SKU.
7. **The `p5e` 3-year reserved rate**, confirmed directly with AWS rather than derived from p5's discount ratio (§8.4).

---

## 9. Bill of materials

| Item | AWS | Qty |
|---|---|---|
| Live LLM fleet + canary | `p5e.48xlarge` (8× H200 141 GB) | **1** (3 GPUs live, 1 canary, 4 spare) |
| Dedicated ML model (scheduled) | `g6.4xlarge` | 1 |
| RL post-training (LoRA) | **Spare H200 on the existing `p5e.48xlarge`** — a second `p5e.48xlarge` only if hard isolation is required | 1 GPU |
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

- [ ] Benchmark **Qwen3.6-35B-A3B on a single H200 at FP8** — confirm ~290–360 tok/s single-stream decode and ~255 KV slots at 8K context
- [ ] **Confirm decode scales ~1.43× over H100 and prefill does not.** The whole §4 latency case rests on decode being bandwidth-bound. If measured decode gain is well below 1.43×, re-run §4 and §8.4 before committing to p5e
- [ ] **Measure p99 end-to-end latency under peak concurrency against the 8 s ceiling** — this is the acceptance criterion, not throughput
- [ ] **Measure the 6-iteration worst case specifically** — modelled at ~7.0 s with ~1 s of margin; this is the tail that used to breach
- [ ] **Verify single-replica operation holds the SLA** (~6.7 s modelled) — it is the basis for treating maintenance and failover as non-SLA events
- [ ] Measure the real prefix-cache hit rate under agentic traffic (assumed ~50% prefill saving)
- [ ] Measure the actual distribution of iterations per conversation (assumed avg 2.5, cap 6) — if the tail is fatter than modelled, §4.1's deadline mechanism carries the SLA
- [ ] Confirm SSE streaming delivers TTFT under ~3 s (modelled ~1.9 s on H200)
- [ ] Load-test knowledge-graph ingestion at 12 writes/s with weight calculation running concurrently on 8 vCPU
- [ ] Rehearse a hot-swap under live traffic — confirm zero dropped connections
- [ ] Rehearse an eval-gate failure — confirm automatic rollback
