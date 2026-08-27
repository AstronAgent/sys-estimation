# Components and system requirements

Every component of the platform, its instance type, and what it needs. One table per tier.

Figures come from [PLATFORM_REPORT.md §9](PLATFORM_REPORT.md) (costs) and [reference/SYSTEM_REQUIREMENTS.md §10](reference/SYSTEM_REQUIREMENTS.md) (bill of materials). Monthly costs are `us-east-1`, 3-year reserved for GPU, on-demand for everything else.

---

## 1. GPU tier — one instance carries the whole LLM platform

**Instance: 1× `p5e.48xlarge`** — 192 vCPU · 2 TB RAM · **8× NVIDIA H200 141 GB HBM3e @ 4.8 TB/s** · $27.344/hr (3-yr reserved) · **$19,961/month**

All 8 GPUs are allocated. Nothing idle.

| # | Component | GPUs | Per-GPU requirement | Notes |
|---|---|---|---|---|
| 1 | SGLang serving — active | 2 | 35 GB weights + 8 GB overhead + KV cache | FP8, TP=1, one GPU per replica |
| 2 | SGLang serving — standby | 1 | same | N+1 failover and hot-swap staging |
| 3 | Production testing canary | 1 | same | Must be H200 to reproduce latency |
| 4 | LoRA RL post-training | 1 | ~66 GB | Off-peak. GPU must be pinned |
| 5 | LoRA continued pre-training | 1 | ~66 GB | Same spare pool, off-peak |
| 6 | Spare / burst | 2 | — | Headroom |

**Serving replica detail:** 35 GB FP8 weights · 48 KiB KV per token · 384 MB per 8K-context stream · ~255 concurrent streams per replica · ~2,900–5,700 tok/s aggregate decode.

**Not feasible on this hardware:** pre-training from scratch (needs ~560 GB and ~2.9 years on one GPU) and full-parameter RL (~205–520 GB). LoRA only.

---

## 2. Mathematical-model workload — its own instance

**Instance: 1× `g6.4xlarge`** — 16 vCPU · 64 GB RAM · **1× NVIDIA L4 24 GB** · $1.323/hr · **$159/month** *(runtime provisional)*

Runs on a schedule, writes to a cache, off the request path. Separate from the serving instance — no GPU contention.

| # | Component | Hardware | Peak memory | Wall-clock |
|---|---|---|---|---|
| 7 | Clustering / whitening / PCA / ICA | CPU (BLAS/LAPACK) | ~400 MB RAM | < 3 s |
| 8 | Causal discovery — LiNGAM, VAR-LiNGAM | CPU | ~160 MB RAM | ~5 s |
| 9 | Causal discovery — DYNOTEARS | CPU, multi-core | ~160 MB RAM | **~10–20 min ← heaviest** |
| 10 | SF-Slinear | CPU | small | ~1 min, or free |
| 11 | ~156 parametric time-series fits | CPU, parallel | small | ~2 min on 16 cores |
| 12 | RNN / GRU / LSTM training | **GPU** | **< 2 GB VRAM** | ~5 min batched |
| | **Total per run** | 16 vCPU + 1 L4 | ~1 GB RAM, < 2 GB VRAM | **~20–30 min** |

The recurrent models are the only VRAM consumers and the reason this stays a GPU instance. Dimensions are still unspecified — see [PLATFORM_REPORT.md §7.6](PLATFORM_REPORT.md).

---

## 3. Application tier — CPU compute

| # | Component | Instance | vCPU | RAM | Qty | $/month |
|---|---|---|---|---|---|---|
| 13 | MCP Graph Orchestrator | `m7g.2xlarge` | 8 | 32 GB | 3 | $714 |
| 14 | Chat API | `m7g.xlarge` | 4 | 16 GB | 2–4 autoscaled | $238 |
| 15 | Auth service | `m7g.large` | 2 | 8 GB | 2 | $120 |
| 16 | Graph ingestion worker | `c7g.2xlarge` | 8 | 16 GB | 1 | $212 |
| 17 | Vector DB (self-hosted) | `r7g.xlarge` | 4 | 32 GB | 3 | $469 |
| | | | | | **Subtotal** | **~$1,912** |

The orchestrator holds the agentic loop and is CPU-bound. The ingestion worker's 8 vCPU supports **incremental** edge-weight updates only — not global graph metrics.

---

## 4. Data and managed services

| # | Component | Configuration | Sizing basis | $/month |
|---|---|---|---|---|
| 18 | PostgreSQL | RDS `db.r7g.2xlarge` Multi-AZ, 8 vCPU / 64 GB, 1 TB gp3 | ~220 GB/yr, ~5 QPS | ~$1,530 |
| 19 | Knowledge graph | Neptune `db.r6g.2xlarge` + read replica, 8 vCPU / 64 GB | ~45 GB/yr, 12 writes/s peak | ~$1,700 |
| 20 | Session cache | ElastiCache Redis `cache.r7g.large` × 2, 2 vCPU / 16 GB | 750 sessions × 200 KB ≈ 150 MB | ~$330 |
| 21 | Observability | CloudWatch + AMP + AMG, OTel traces | per-token latency, KV use, cache hits | ~$300 |
| 22 | Object storage | S3, 5 TB | 35 GB/checkpoint × 50 retained ≈ 2 TB | ~$115 |
| 23 | Load balancer | ALB + AWS WAF | ~3 req/s peak | ~$30 |
| 24 | ML result cache | DynamoDB, on-demand, native TTL | small at any reading | ~$25 |
| 25 | Data transfer out | ~120 MB/day + overhead | — | ~$20 |
| 26 | Task queue | SQS | 75k messages/day | ~$1 |
| | | | **Subtotal** | **~$4,051** |

---

## 5. LLM model — requirements in detail

**Qwen/Qwen3.6-35B-A3B** — Mixture-of-Experts: **35B total parameters, ~3B active per token.**

| Property | Value | Why it matters |
|---|---|---|
| Total parameters | 35B | **Sets VRAM** — 35 GB at FP8 |
| Active parameters | ~3B | **Sets compute and bandwidth** — ~3 GB read per token |
| Layers | 48 | KV cache size |
| KV heads | 4 | KV cache size |
| Head dimension | 128 | KV cache size |
| Precision | **FP8** | BF16 fits on 141 GB but halves decode speed |
| Context window | 8,192 tokens | 384 MB of KV per active stream |

| Requirement | Value |
|---|---|
| **GPU** | 1× H200 141 GB per replica, **TP=1** — no multi-GPU sharding needed |
| **VRAM per replica** | 35 GB weights + 8 GB runtime overhead + KV cache |
| **KV cache** | 48 KiB/token → 384 MB per 8K stream → **~255 concurrent streams** |
| **Memory bandwidth** | 4.8 TB/s — **this is the binding hardware property, not FLOPS** |
| **Decode throughput** | ~290–360 tok/s single-stream · ~260 tok/s at batch 12 · ~2,900–5,700 tok/s aggregate |
| **Prefill throughput** | ~20,000–30,000 tok/s — compute-bound, **does not improve on H200** |
| **Runtime** | SGLang with RadixAttention prefix caching |
| **Load balancer** | Least-outstanding-requests with **prefix-hash affinity** — required, not optional |

**Precision options, and why FP8 wins:**

| Precision | Weights | Streams @ 8K | Verdict |
|---|---|---|---|
| BF16 | ~70 GB | ~164 | Fits, but 2× bytes/token halves decode → ~8.5 s end-to-end. Rejected |
| **FP8** | **~35 GB** | **~255** | **Recommended** |
| INT4 | ~18 GB | ~300 | Cost tier — benchmark quality first |

**Training requirements on this model:**

| Task | Memory | Fits on 1× H200? |
|---|---|---|
| LoRA RL (GRPO/RLOO) | ~66 GB | ✓ with 2× headroom |
| QLoRA (NF4 base) | ~48 GB | ✓ |
| Full-parameter FP8 + 8-bit Adam | ~205 GB | ✗ |
| Full-parameter BF16 + AdamW FP32 | ~520 GB | ✗ |
| Pre-training from scratch | ~560 GB, ~2.9 years | ✗ **not feasible** |

LoRA targets the attention projections (q, k, v, o). Adapting all expert FFNs is expensive and rarely needed; the router can be adapted but destabilises easily.

---

## 6. Mathematical models — every model and what it needs

All of these run on the `g6.4xlarge` of §2, on a schedule, off the request path.

**Symbols:** `n` observations · `d` raw series · `k` retained components · `p` lag order · `M` ≈ 156 fitted models · `c` clusters · `Θ` neural parameters.

### 6.1 Preprocessing stages

| Model | Complexity | Hardware | Memory |
|---|---|---|---|
| Clustering (k-means / hierarchical) | `O(i·n·c·d)` | CPU | small |
| Whitening | `O(n·d² + d³)` | CPU | `O(d²)` covariance |
| PCA | same eigendecomposition as whitening — **counted once** | CPU | `~8·n·d` bytes |
| ICA (FastICA / Infomax) | `O(j·n·k²)` | CPU | small |

### 6.2 Causal discovery — cubic in the variable count, and does not parallelise

| Model | Complexity | Hardware | Notes |
|---|---|---|---|
| **ICA-LiNGAM** | `O(j·n·d²)` + `O(d³)` | CPU | Reuses ICA. **Not deterministic** — local optima |
| **DirectLiNGAM** | `~O(n·d³)`; kernel measures → `O(n·d³M² + d⁴M³)` | CPU | Deterministic. Preferred below `d` ≈ 100 |
| **VAR-LiNGAM** | `O(n·d²p² + (d·p)³)` + LiNGAM on residuals | CPU | Lag order enters **cubically** |
| **DYNOTEARS** | `O(T_out·T_in·(n·d²(1+p) + d³))` | CPU, multi-core | **The expensive one** — thousands of iterations |
| **SF-Slinear** | `O(d²·n·s)` *or* negligible | CPU | **Name ambiguous — needs confirming** |

**The requirement that matters here is dimensional, not hardware.** These are one joint estimation over all variables, so adding cores does not help. Run them on the `k` retained components — on raw `d` = 500 series, DYNOTEARS goes from ~20 minutes to hours.

**LiNGAM has a data requirement, not just a compute one:** the noise must be **non-Gaussian**. Under Gaussian noise the causal ordering is unidentifiable and LiNGAM returns an arbitrary answer rather than an error. This needs an explicit check.

### 6.3 Parametric time-series models (~156 fits)

| Family | Fit cost driver | Parallel? |
|---|---|---|
| ARIMA / SARIMA | optimiser iterations × `n` | ✓ embarrassingly |
| GARCH / EGARCH / GJR | same, but slow to converge | ✓ |
| Exponential smoothing (ETS) | cheap, near closed-form | ✓ |
| State-space / Kalman | `O(n · state_dim³)` per likelihood evaluation | ✓ |
| **VAR / VECM** | **`O(n·k²p)` + `O(k³p³)` joint solve** | **✗ does not decompose** |

**~156 univariate fits ≈ 2 minutes on 16 cores. One VAR over 156 series is a different problem entirely.** Which it is has not been confirmed.

### 6.4 Neural sequence models — the only GPU stage

| Model | Parameters per layer | Relative cost |
|---|---|---|
| **RNN** | `h·(x + h) + h` | 1× |
| **GRU** | `3·(h·(x + h) + h)` | 3× |
| **LSTM** | `4·(h·(x + h) + h)` | 4× |

| Requirement | Value |
|---|---|
| Training compute | `6 · Θ · tokens_seen` — the same rule as the LLM |
| **GPU** | 1× L4 24 GB (or smaller) |
| **VRAM** | **< 2 GB** at a representative shape — against 16 GB requested |
| Activation memory (BPTT) | `B · L_seq · h · G · n_layers · bytes` — ~34 MB at batch 256 |
| Wall-clock | ~5 minutes for 156 models, batched |
| **Utilisation** | **single-digit % MFU** — timesteps are strictly sequential |

**The requirement is batching, not a faster card.** Each timestep waits for the previous one, so a bigger GPU runs at the same few percent of a larger number. Batch across the 156 models — it is free.

**Sensitivity:** hidden width enters quadratically (128 → 512 is 16×), sequence length linearly in both compute and memory (64 → 512 is 8× each).

---

## 7. MCP harness — requirements

The orchestrator holds the agentic loop. The GPUs only decode.

| Requirement | Specification |
|---|---|
| **Compute** | 3× `m7g.2xlarge` — 8 vCPU / 32 GB each. CPU-bound, no GPU |
| **Topology** | MCP servers are **graph nodes**; **edges define allowed call transitions** |
| **Tool dispatch** | Route through the graph — **not** a flat tool list exposed to the model |
| **Parallelism** | Independent tools modelled as parallel-dispatchable edges — saves ~0.8 s per avoided pass |
| **Queue** | SQS, decouples long agentic runs from the HTTP tier |
| **Streaming** | SSE on the Chat API — TTFT ~1.9 s |

**Two auth layers, both required:**

| Layer | Mechanism |
|---|---|
| User-facing | OIDC/OAuth2 + JWT with refresh, per-user rate limits |
| Service-to-service | **Per-tool scoped tokens** |

> The LLM must never hold one credential that reaches every backend. A prompt-injected tool call would otherwise reach the entire data tier.

**Three cost and latency guards, all required — they are not alternatives:**

| Guard | Value | What it bounds |
|---|---|---|
| Iteration cap | 6 passes | Cost, **not latency** |
| Per-conversation token budget | — | Cost |
| **Wall-clock deadline** | **6 s** | **Latency — the only hard guarantee** |

At 6 seconds elapsed the orchestrator stops issuing tool calls and forces a final answer from what it has.

---

## 8. Graph engine — requirements

Full knowledge graph, continuously ingested from conversations.

| Requirement | Specification |
|---|---|
| **Engine** | Neptune `db.r6g.2xlarge` + read replica, or Neo4j on `r7g.2xlarge` + replica |
| **Ingestion worker** | `c7g.2xlarge` — **8 vCPU** / 16 GB, dedicated to edge-weight calculation |
| **Write rate** | 3.5 nodes/s average, **~12/s peak** |
| **Query latency budget** | ~0.15 s per tool round — inside the §7 latency budget |

**Growth**, assuming ~10 entities and ~20 edges extracted per conversation *(extraction density needs confirming)*:

| Metric | Per day | Per year |
|---|---|---|
| Nodes | 300,000 | ~110 M |
| Edges | 600,000 | ~220 M |
| Storage with indexes | ~120 MB | **~45 GB** |

**Storage and write rate are both comfortable. The constraint is the weight calculation:**

| Weight computation | 8 vCPU sufficient? |
|---|---|
| **Incremental / windowed** edge weights on a 12 writes/s stream | **✓ ample** |
| **Global** metrics — PageRank, centrality, community detection over 110M nodes | **✗ hours on 8 cores** — needs Neptune Analytics or Spark on EMR |

> **A continuously ingesting graph with no retention policy grows without bound.** Decay, pruning, or entity consolidation is needed from day one — retrofitting it onto a 200M-edge graph is painful.

---

## 9. Totals

| Tier | $/month | $/year |
|---|---|---|
| GPU (3-yr reserved) | $19,961 | $239,533 |
| Non-GPU compute | $1,912 | $22,944 |
| Managed services | $4,051 | $48,612 |
| **All-in** | **$25,924** | **$311,089** |

Per conversation **$0.0284** · per user per month **$8.64** · at 30,000 conversations/day.

---

## 10. Software stack

| Layer | Choice |
|---|---|
| Model | Qwen/Qwen3.6-35B-A3B — MoE, 35B total / ~3B active |
| Serving runtime | SGLang, FP8, TP=1, RadixAttention prefix caching |
| Weight hot-swap | SGLang `update_weights_from_disk`, blue/green fallback |
| Training | LoRA on attention projections (q, k, v, o); GRPO/RLOO for RL |
| Orchestration | MCP servers as graph nodes, edges as allowed call transitions |
| Auth | OIDC/OAuth2 + JWT (users); per-tool scoped tokens (services) |
| Analytics | LAPACK/BLAS for classical stages; GPU framework for RNN/GRU/LSTM |

---

## 11. Hard constraints

| Constraint | Value | Status |
|---|---|---|
| Response time SLA | 2–8 seconds | ✓ ~4.6 s typical |
| 6-iteration worst case | ≤ 8 s | ✓ ~7.0 s — **~1 s margin only** |
| Single-replica degraded mode | ≤ 8 s | ✓ ~6.7 s |
| Memory per replica | ≤ 141 GB | ✓ ~47.6 GB at batch 12 |
| Peak decode throughput | ≥ ~1,200 tok/s | ✓ ~2.4× over |
| Agentic iteration cap | 6 passes | Enforced in orchestrator |
| Wall-clock deadline | 6 s | The only hard guarantee against the ceiling |

**Three guards are all required, not alternatives:** iteration cap, per-conversation token budget, and wall-clock deadline. An iteration cap does not bound latency; a deadline does.
