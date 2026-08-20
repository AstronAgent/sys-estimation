# Qwen3.6-35B-A3B Platform — AWS Deployment Architecture

SGLang · graph MCP harness · continuous RL hot-swap — 3k DAU / ~750 concurrent / 30k conversations per day / 2–8 s response SLA

> **A3B = Mixture-of-Experts: 35B total parameters, ~3B active per token.**
> VRAM is sized by total params (~35 GB at FP8) but compute and memory bandwidth by active params — so decode runs ~4–5× faster than a dense 35B and each replica fits on **one** H100 at TP=1.
> Live fleet drops from 6× H100 to **3× H100**. Replica count is set by the **latency SLA, not throughput**.

*Markdown version of `deployment-architecture.html`, which carries the same figures as inline SVG with PNG/PDF export.*
*Full AWS cost breakdown in `PLATFORM_REPORT.md`; engineering detail in `SYSTEM_REQUIREMENTS.md`.*

---

## Fig. 1 — Live production request path (AWS)

All components sit in an **AWS VPC on private subnets, with no public GPU ingress**. Dashed groups below are security-group boundaries.

```mermaid
flowchart TB
    users["End Users<br/>3k DAU · ~750 concurrent sessions"]
    alb["ALB + AWS WAF<br/>L7 · TLS termination · rate limit<br/>peak ~3 req/s — trivial load"]
    auth["Auth Service<br/>Cognito / OIDC · JWT<br/>2× m7g.large"]
    api["Chat API — stateless<br/>SSE stream · TTFT under 3 s<br/>2–4× m7g.xlarge autoscaled"]
    redis["ElastiCache Redis<br/>sessions · tool-call state<br/>~150 MB · cache.r7g.large + replica"]
    sqs[["SQS async task queue"]]
    mcp["MCP Graph Orchestrator<br/>tools as graph nodes · independent edges in parallel<br/>guards: 6-iteration cap + token budget + 6 s deadline<br/>3× m7g.2xlarge — holds the loop, not the GPU"]
    ilb["Inference Load Balancer<br/>least-outstanding-requests<br/>prefix-hash affinity → RadixAttention hits"]

    subgraph gpupool["sg-inference :30000 — orchestrator only · p5.48xlarge"]
        s1["SGLang 1 · ACTIVE<br/>35B-A3B MoE · FP8 · TP=1<br/>1× H100 80G · ~96 KV slots"]
        s2["SGLang 2 · ACTIVE<br/>35B-A3B MoE · FP8 · TP=1<br/>1× H100 80G · ~96 KV slots"]
        s3["SGLang 3 · STANDBY<br/>N+1 spare + hot-swap target<br/>1× H100 80G"]
    end

    subgraph toolsvc["sg-tools — per-tool scoped tokens, no shared superuser credential"]
        pg["RDS PostgreSQL<br/>users · auth · chat history<br/>~220 GB/yr · db.r7g.2xlarge Multi-AZ"]
        vec["Vector DB<br/>OpenSearch / Qdrant · RAG<br/>~500 GB/yr with index · 3 nodes"]
        kg["Knowledge Graph<br/>Neptune / Neo4j r7g.2xlarge<br/>~110M nodes/yr · ~45 GB/yr"]
        mlc["ML Result Cache<br/>DynamoDB + native TTL<br/>read-only for the LLM · ~ms"]
    end

    subgraph offline["OFFLINE / SCHEDULED — never on the request path"]
        giw["Graph Ingestion Worker<br/>continuous · edge-weight calc<br/>c7g.2xlarge — 8 vCPU · ~12 writes/s"]
        sml["Scheduled ML Model<br/>runs on a fixed frequency<br/>g6.4xlarge — 16 vCPU / 64 GB / L4"]
    end

    users -->|"1 · HTTPS / SSE stream"| alb
    alb -->|"2"| api
    api -.->|"3 · verify JWT"| auth
    api -->|"4 · session state"| redis
    api -->|"5 · async dispatch"| sqs
    sqs --> mcp
    mcp <-->|"6 · agentic loop, 2–6 passes per conversation"| ilb
    ilb -->|"7 · least-outstanding + prefix affinity"| s1
    ilb --> s2
    ilb --> s3
    mcp -->|"8 · MCP tool calls"| toolsvc
    giw -.-> kg
    sml -.-> mlc

    classDef svc fill:#064e3b,stroke:#34d399,stroke-width:1.5px,color:#e2e8f0
    classDef db fill:#4c1d95,stroke:#a78bfa,stroke-width:1.5px,color:#e2e8f0
    classDef lb fill:#78350f,stroke:#fbbf24,stroke-width:1.5px,color:#e2e8f0
    classDef sec fill:#881337,stroke:#fb7185,stroke-width:1.5px,color:#e2e8f0
    classDef cache fill:#083344,stroke:#38bdf8,stroke-width:1.5px,color:#e2e8f0
    classDef queue fill:#7c2d12,stroke:#fb923c,stroke-width:1.5px,color:#e2e8f0
    classDef job fill:#1e293b,stroke:#64748b,stroke-width:1.5px,color:#e2e8f0
    classDef ext fill:#1e293b,stroke:#94a3b8,stroke-width:1.5px,color:#e2e8f0

    class users ext
    class alb,ilb lb
    class auth sec
    class api,mcp,s1,s2,s3 svc
    class redis,mlc cache
    class sqs queue
    class pg,vec,kg db
    class giw,sml job
```

### Flow

| # | Step | Notes |
|---|---|---|
| 1 | Users → ALB | HTTPS, SSE streaming back |
| 2 | ALB → Chat API | ~3 req/s peak — trivial for the web tier |
| 3 | Chat API → Auth | JWT verification |
| 4 | Chat API → Redis | Session and tool-call state |
| 5 | Chat API → SQS → Orchestrator | Decouples long agentic runs from the HTTP tier |
| 6 | **Orchestrator ↔ Inference LB** | **The agentic loop — traversed 2–6 times per conversation** |
| 7 | Inference LB → SGLang pool | Least-outstanding-requests + prefix-hash affinity |
| 8 | Orchestrator → MCP tool servers | Per-tool scoped tokens |
| 9 | Hot-swap weight update → SGLang pool | See Fig. 2. Four of the eight GPUs remain free. |

### Component specifications

| Component | Instance / service | Sizing basis |
|---|---|---|
| Edge load balancer | ALB + AWS WAF | TLS, rate limiting; ~3 req/s peak |
| Auth service | Cognito, or 2× `m7g.large` | OIDC / OAuth2, JWT + refresh |
| Chat API | 2–4× `m7g.xlarge` | Stateless, SSE, autoscaled |
| MCP Graph Orchestrator | 3× `m7g.2xlarge` (8 vCPU / 32 GB) | CPU-bound routing; holds the loop, not the GPU |
| Async queue | SQS | 75k messages/day |
| Session cache | ElastiCache `cache.r7g.large` + replica | 750 sessions × ~200 KB ≈ 150 MB |
| **SGLang pool** | **3× H100 80 GB on one `p5.48xlarge`** | **2 active + 1 standby, FP8, TP=1** |
| Inference LB | Least-outstanding + prefix hash | Load-bearing, not an optimisation |
| Chat history / auth | RDS `db.r7g.2xlarge` Multi-AZ, 1 TB gp3 | ~220 GB/yr, ~5 QPS — no sharding |
| Vector DB | OpenSearch Serverless, or 3× `r7g.xlarge` | ~500 GB/yr with index overhead |
| Knowledge graph | Neptune `db.r6g.2xlarge`, or Neo4j on `r7g.2xlarge` | ~110 M nodes/yr, ~45 GB/yr |
| Graph ingestion worker | `c7g.2xlarge` (8 vCPU) | Edge-weight calculation, ~12 writes/s |
| ML result cache | DynamoDB + native TTL | Read-only for the LLM |
| Scheduled ML model | `g6.4xlarge` (16 vCPU / 64 GB / L4) | Fixed-frequency job, EventBridge-triggered |

---

## Fig. 2 — Training, gated hot-swap, and the scheduled ML track

```mermaid
flowchart LR
    pre["1 · Pre-training<br/>LoRA continued / domain-adaptive<br/>~66 GB → fits 1× H200 or B200<br/>H200 ~1.9B tok/day · B200 ~4.3B<br/>from scratch: 1.3–2.9 yr — not feasible"]
    rl["2 · Post-training, RL<br/>GRPO · LoRA · reference model is free<br/>~66 GB → 1× H200 / B200 / H100<br/>~600–1,200 RL steps/day on H200<br/>full-param ~520 GB — will not fit"]
    reg["3 · Model Registry<br/>S3 · versioned, immutable<br/>35 GB per FP8 checkpoint<br/>retain 50 → ~2 TB"]
    test["4 · Production Testing<br/>SGLang canary · shadow traffic<br/>1 GPU on the same p5.48xlarge<br/>must be H100 — L40S cannot<br/>reproduce production latency"]
    gate{"Automated Eval Gate<br/>safety · regression · task accuracy<br/>p99 vs the 8 s SLA · tool validity<br/>fail → block + auto-rollback"}
    live["5 · Live SGLang Fleet — Fig. 1<br/>hot-swap: update_weights_from_disk<br/>in-place, connections held open<br/>fallback: blue/green drain via inference LB"]

    pre -->|"1"| rl
    rl -->|"2"| reg
    reg -->|"3"| test
    test -->|"4 · eval"| gate
    gate -->|"5 · promote"| live
    live -.->|"6 · prod traces + reward signal → RL replay buffer"| rl

    classDef svc fill:#064e3b,stroke:#34d399,stroke-width:1.5px,color:#e2e8f0
    classDef db fill:#4c1d95,stroke:#a78bfa,stroke-width:1.5px,color:#e2e8f0
    classDef lb fill:#78350f,stroke:#fbbf24,stroke-width:1.5px,color:#e2e8f0
    classDef sec fill:#881337,stroke:#fb7185,stroke-width:1.5px,color:#e2e8f0

    class pre lb
    class rl,test,live svc
    class reg db
    class gate sec
```

> **Promotion gate — there is no direct RL → production path.** Every checkpoint clears the automated eval suite before it reaches live traffic. A continuous training pipeline that can push straight to production is a continuous outage pipeline.

### Offline / scheduled track

Runs on a fixed frequency and is **never on the request path**.

```mermaid
flowchart LR
    ev(["EventBridge schedule"])
    sml["Dedicated ML Model<br/>Spot-eligible<br/>g6.4xlarge — 16 vCPU / 64 GB / 24 GB VRAM"]
    cache["ML Result Cache<br/>DynamoDB + native TTL"]
    mcp["MCP tool read — Fig. 1<br/>single-digit ms"]

    ev --> sml -->|"7"| cache --> mcp

    classDef job fill:#1e293b,stroke:#64748b,stroke-width:1.5px,color:#e2e8f0
    classDef cch fill:#083344,stroke:#38bdf8,stroke-width:1.5px,color:#e2e8f0
    class ev,sml job
    class cache,mcp cch
```

**Open on this track:** run frequency, maximum acceptable staleness, and cache-miss policy — serve stale, trigger an on-demand run, or return unavailable. The last is usually correct for a latency-bound system.

---

## MoE halves the fleet

- VRAM by **total** params: ~35 GB at FP8 → fits **1× H100, TP=1**
- Bandwidth by **active** params: 3 GB per token, not 35
- Decode **~4–5× faster** than a dense 35B — ~200 tok/s single-stream
- KV ~48 KB/token → **~96 slots** per replica at 8K context
- **BF16 would not fit** (70 GB weights leaves ~2 GB for KV) — FP8 is required for TP=1

## Latency sets the replica count

- Throughput needs **1 replica**; the 8-second SLA needs **2**
- 1 replica → batch 25 → 120 tok/s per stream → **~8.5 s, breaches**
- 2 replicas → batch 12 → 180 tok/s per stream → **~6.4 s, meets**
- **The 6-iteration cap breaches 8 s** (~9.6 s) — this needs a wall-clock deadline, not just an iteration cap
- SSE streaming brings TTFT to ~2.6 s — the highest-leverage fix

## Sizing basis

- 30k conversations/day × 2.5 passes = **75k inference calls/day**
- Decode **30 M tokens/day** → ~1,200 tok/s peak
- Prefill 306 M uncached → **~150 M with prefix cache**
- Little's Law: **~25 streams generating**, not 750 — the session count never drove GPU sizing
- Prefix affinity at the inference LB is **load-bearing**, not an optimisation

## AWS footprint

- **1× `p5.48xlarge`** carries live (3 GPUs) + canary (1 GPU), 4 spare for training
- Price against **Capacity Blocks / Savings Plans** — on-demand p5 is the worst rate
- `g6.4xlarge` for the scheduled ML model — an exact match to the stated spec
- `c7g.2xlarge` is the 8 vCPU graph weight worker
- Data tier is small: ~220 GB/yr Postgres, ~500 GB/yr vector, ~45 GB/yr graph

## Training on one GPU — verdict

- **LoRA RL fits: ~66 GB** on 1× H200 (141 GB) or B200 (180 GB). ~600–1,200 steps/day
- **Full-parameter RL does not: ~520 GB.** AdamW FP32 state alone is 280 GB
- With LoRA the KL **reference model is free** — base weights, adapters disabled
- LoRA continued pre-training viable: **~1.9 B tok/day on H200, ~4.3 B on B200**
- **From-scratch pre-training: 1.3–2.9 years on one GPU.** Not feasible

## Open risks

- **AWS sells no single H200 or B200** — both are 8-GPU SKUs. Use the **4 spare H100s on the `p5` already in the plan** rather than buying another instance
- Training and serving would then share a machine — pin GPUs so a training job cannot breach the 8 s SLA
- Graph weight calculation: 8 vCPU is fine **incrementally**, not for global PageRank over 110 M nodes
- Continuous graph ingestion needs a **retention / decay policy** from day one
- **`p5e` (H200) is worth pricing for serving too** — ~1.43× decode, ~5.2 s end-to-end

---

*Qwen/Qwen3.6-35B-A3B on SGLang · graph MCP harness · AWS · continuous RL with gated hot-swap.*
*Directional estimates. Benchmark the real model on a single H100 and measure p99 against the 8-second ceiling before procurement.*
*Full cost breakdown in `PLATFORM_REPORT.md` — $279,842/yr on a 3-year reserved `p5.48xlarge`, or $0.0256 per conversation.*
