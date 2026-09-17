# Databricks Scientific RAG & LLMOps Benchmark Platform

## Why this project exists

Building a RAG demo is easy. Building a RAG system that can answer **how well retrieval works, how much latency each stage adds, what happens when the corpus is real, and how the LLM path is governed and observable** is much harder.

This project was built to answer that engineering question.

The goal was not simply to connect a vector database to an LLM. The goal was to build a **measurable, reproducible Databricks-native RAG platform** that demonstrates the complete lifecycle:

1. ingest and validate a real scientific corpus,
2. transform it into retrieval-ready chunks,
3. create and operate a managed vector/hybrid search index,
4. build a repeatable retrieval evaluation dataset,
5. compare retrieval strategies with proper ranking metrics,
6. send retrieved evidence through a governed LLM endpoint,
7. trace the complete request from retrieval through generation,
8. capture usage, latency, token, and governance evidence,
9. distinguish development plumbing from real-data benchmark claims.

The result is a portfolio project that demonstrates **data engineering, information retrieval, GenAI application engineering, LLMOps, observability, governance, and performance evaluation** in one system.

---

## What problem does it solve?

A production RAG system has several independent failure modes:

- the source data may be incomplete or poorly chunked,
- embeddings may be expensive or slow to generate,
- vector search may retrieve semantically similar but irrelevant content,
- keyword-heavy queries may perform differently from semantic queries,
- an LLM may answer fluently even when retrieval is weak,
- latency can come from retrieval, prompt construction, or generation,
- evaluation metrics can be implemented incorrectly,
- model calls require governance, rate limits, and usage tracking,
- development results on synthetic data can look unrealistically perfect.

This project makes those concerns measurable instead of hiding them behind a single demo question.

---

## What was built

### Data layer

A real arXiv-derived scientific corpus was ingested into Databricks and stored in Unity Catalog / Delta Lake.

Measured real-data baseline:

| Item | Measured value |
|---|---:|
| Scientific documents | **10,000** |
| Generated chunks | **11,528** |
| Documents without chunks | **0** |
| Empty chunks | **0** |
| Duplicate chunk IDs | **0** |
| Average tokens/chunk | **110.35** |
| P50 tokens/chunk | **104** |
| P95 tokens/chunk | **192** |

The raw corpus was also staged in a Unity Catalog Volume so ingestion, Delta transformation, and search indexing were separated cleanly.

### Retrieval layer

Two AI Search indexes were used for different purposes:

- **28,000-chunk synthetic development index** — plumbing validation only.
- **11,528-chunk real arXiv index** — final measured retrieval and RAG evidence.

The production index uses:

- Databricks AI Search
- Delta Sync
- triggered synchronization
- managed embeddings
- `databricks-qwen3-embedding-0-6b`
- `chunk_id` as the primary key

This development-to-real-data progression was deliberate: first prove the integration path cheaply, then move benchmark claims onto real data.

### Evaluation layer

A deterministic 500-query retrieval evaluation set was generated from the real 10K-document corpus.

Each query has:

- a unique query ID,
- query text derived from a real abstract,
- one known relevant `document_id`,
- a reproducible source/methodology.

The evaluation set contains:

| Item | Value |
|---|---:|
| Queries | **500** |
| Distinct relevant documents | **500** |
| Empty queries | **0** |

This is a reproducible retrieval benchmark, not a claim of human-judged semantic relevance. That distinction matters.

### Generation layer

Retrieved chunks are passed to a Unity Gateway model service:

```text
p14_rag_llmops.default.p14-rag-llm-dev
```

The model service routes to:

```text
system.ai.gpt-oss-20b
```

The final RAG implementation:

- performs HYBRID top-k retrieval,
- constructs a context-constrained prompt,
- instructs the model not to invent unsupported facts,
- returns chunk citations,
- filters reasoning-model internals and exposes only final `output_text`,
- records an MLflow trace across the complete request.

---

## Architecture

```text
                        ┌──────────────────────────────┐
                        │ Real scientific corpus       │
                        │ common-pile/arxiv_abstracts  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ Unity Catalog Volume         │
                        │ p14_stage                    │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │ Delta Lake                              │
                  │ scientific_documents_prod              │
                  │ scientific_chunks_prod                 │
                  │ retrieval_eval_prod                    │
                  └───────────────────┬─────────────────────┘
                                      │
                                      ▼
                        ┌──────────────────────────────┐
                        │ Databricks AI Search         │
                        │ ANN / HYBRID                 │
                        │ 11,528 indexed chunks        │
                        └──────────────┬───────────────┘
                                       │ top-k chunks
                                       ▼
                        ┌──────────────────────────────┐
                        │ Grounded prompt builder      │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ Unity Gateway Model Service  │
                        │ GPT OSS 20B                  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ Grounded scientific answer   │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
              ┌────────────────────────────────────────────────┐
              │ MLflow tracing + Unity Gateway usage tracking  │
              │ retrieval | prompt | generation | tokens       │
              └────────────────────────────────────────────────┘
```

---

## What did the project achieve?

### 1. It proved the complete Databricks-native RAG path

The final request successfully executed:

```text
question
  -> HYBRID AI Search
  -> top-5 real scientific chunks
  -> grounded prompt construction
  -> Unity Gateway model service
  -> final grounded answer
  -> MLflow trace
```

Final real-data RAG measurement:

| Measurement | Result |
|---|---:|
| End-to-end latency | **2.55 s** |
| Input tokens | **1,219** |
| Output tokens | **305** |
| Total tokens | **1,524** |
| Result | `RAG_REAL_E2E_FINAL_ONLY=PASS` |

MLflow decomposed the request into independently observable spans:

```text
project14_real_rag_final_only
├── retrieve_hybrid
├── build_grounded_prompt
└── generate_answer
```

This makes the application debuggable rather than treating an LLM request as one opaque operation.

### 2. It measured ANN versus HYBRID search on real data

Corrected 500-query results:

| Metric | ANN | HYBRID |
|---|---:|---:|
| Hit@5 | 0.984 | **0.996** |
| Hit@10 | 0.990 | **0.998** |
| MRR | 0.9568 | **0.9821** |
| NDCG@10 | 0.9651 | **0.9860** |
| P50 latency | **469.81 ms** | 479.31 ms |
| P95 latency | 630.37 ms | **615.64 ms** |
| Failures | 0 | 0 |

### What those results demonstrate

For this corpus and evaluation method:

- HYBRID returned a relevant document in the top 5 for **99.6%** of queries.
- HYBRID improved MRR from **0.9568 to 0.9821**, meaning relevant documents tended to appear earlier.
- HYBRID improved NDCG@10 from **0.9651 to 0.9860**.
- The P50 latency difference was only about **9.5 ms** in this run.
- HYBRID actually produced a slightly lower measured P95 than ANN in this run.
- Both modes completed all 500 queries with **zero request failures**.

These are workload-specific measurements, not universal claims about Databricks.

### 3. It caught and corrected an evaluation bug

The first NDCG implementation produced values greater than 1 because multiple chunks from the same relevant document were being counted as multiple relevance gains.

That result was rejected rather than published.

The benchmark was recomputed from saved query results without rerunning the expensive searches, using the first rank of the one known relevant document per query.

That correction is important because the project demonstrates not just running metrics, but validating whether the metrics themselves are meaningful.

### 4. It measured managed embedding behavior

The real index initially reported approximately **1 row/s** during startup, then accelerated to roughly **5 rows/s** later in the initial synchronization.

This demonstrated that early ETA/throughput observations were not representative of sustained managed embedding throughput.

It also exposed an important scaling consideration: for much larger corpora, precomputed/batch embeddings may be preferable to relying solely on managed per-index embedding generation.

### 5. It added LLM governance and observability

The Unity Gateway model service was configured with:

- usage tracking,
- inference-table configuration,
- request/token rate limits,
- governed Unity Catalog model-service access.

`system.ai_gateway.usage` confirmed real requests were recorded and routed through the model service.

### 6. It tested a capability that was not available in the workspace

A managed reranker was explicitly tested.

The API returned a workspace-level configuration restriction preventing access to the reranker model. The project therefore documents reranking as **tested but unavailable in this environment** instead of fabricating results or silently skipping the feature.

---

## What this project demonstrates technically

### Data engineering

- external dataset ingestion
- resilient API fetching with resume/retry behavior
- Unity Catalog Volumes
- Delta tables
- schema validation
- change data feed configuration
- chunk-generation pipelines
- deterministic evaluation-data generation

### Information retrieval

- ANN vector retrieval
- hybrid semantic + lexical retrieval
- Hit@K
- MRR
- NDCG@10
- latency percentile analysis
- known-relevant-document evaluation
- duplicate-chunk metric handling

### GenAI / RAG engineering

- context retrieval
- grounded prompt construction
- evidence citations
- reasoning-model response parsing
- final-output filtering
- end-to-end RAG execution

### LLMOps

- MLflow experiments
- MLflow GenAI tracing
- nested retrieval/generation spans
- artifact logging
- Unity Gateway model services
- token/request telemetry
- inference logging configuration
- rate limiting

### Engineering discipline

- synthetic plumbing validation separated from real-data claims
- measured evidence instead of hard-coded benchmark claims
- explicit environment limitations
- corrected metric implementation after validation
- cost-aware reuse of endpoints and saved benchmark results

---

## Development evidence versus final evidence

The synthetic development corpus intentionally produced unusually easy retrieval queries and perfect retrieval scores.

Those results are **not** used as production-quality evidence.

The final README claims are based on:

```text
10,000 real arXiv documents
11,528 real chunks
500 real-corpus deterministic retrieval queries
real Databricks AI Search index
real Unity Gateway generation calls
```

---

## Evidence

Screenshots are under:

```text
docs/evidence/ui/
```

Important final evidence:

```text
ai_search_real_index_online_11528.png
ai_search_real_initial_sync_completed.png
mlflow_real_ai_search_500_corrected_ann_metrics.png
mlflow_real_ai_search_500_corrected_quality_metrics.png
mlflow_real_ai_search_500_corrected_hybrid_metrics.png
mlflow_real_rag_end_to_end_trace.png
unity_gateway_usage_tracking_rag_requests.png
ai_playground_grounded_rag_example.png
unity_gateway_p14_rag_model_service.png
unity_gateway_model_service_governance.png
unity_gateway_inference_table_setup.png
unity_gateway_rate_limits_setup.png
```

Superseded screenshots are retained separately under:

```text
docs/evidence/archive/
```

They are kept for audit/history but should not be used for final benchmark claims.

---

## Scope and limitations

This project deliberately avoids overstating what was measured.

### Measured

- 10K real documents
- 11,528 real chunks
- 500-query real-corpus retrieval benchmark
- ANN and HYBRID retrieval
- real end-to-end RAG
- MLflow tracing
- Unity Gateway usage telemetry

### Designed for, but not claimed as executed

The architecture was designed so the corpus can be expanded toward:

```text
100K -> 500K -> 1M documents
```

Those scales were **not executed as part of the final measured evidence**, so this repository does not claim million-document benchmark results.

---

## Repository structure

```text
config/                 project configuration
data/                   local development data (gitignored where appropriate)
docs/evidence/          screenshots and methodology evidence
notebooks/              Databricks/MLflow setup examples
results/                measured benchmark outputs
scripts/                ingestion, chunking, evaluation, RAG, tracing
sql/                    Unity Catalog / Delta setup
README.md               project narrative and measured results
```

---

## Reproducibility

Core sequence:

```bash
python scripts/validate_arxiv_corpus.py \
  --file data/arxiv_real_10k.jsonl \
  --expected-rows 10000

python scripts/load_real10k_delta.py
python scripts/build_real10k_chunks.py
python scripts/validate_real10k_delta.py

python scripts/build_real_eval_500.py
python scripts/upload_real_eval_to_volume.py
python scripts/load_real_eval_delta.py

python scripts/benchmark_real_ai_search_500.py
python scripts/recompute_real_metrics_corrected.py

python scripts/rag_real_e2e.py
```

Databricks resources such as the AI Search index and Unity Gateway model service must exist before their corresponding benchmark steps are run.

---

## Security

Credentials are loaded from `.env` and must not be committed.

Expected variables:

```text
DATABRICKS_SERVER_HOSTNAME
DATABRICKS_HTTP_PATH
DATABRICKS_TOKEN
DATABRICKS_CATALOG
DATABRICKS_SCHEMA
HF_TOKEN
```

Always verify `.env` is ignored before publishing.

---

## Bottom line

This project demonstrates how to move from a simple RAG proof of concept to an **evaluated and observable retrieval system**:

- the data was validated,
- retrieval was benchmarked,
- ranking metrics were checked and corrected,
- real scientific content replaced synthetic benchmark claims,
- generation was grounded in retrieved evidence,
- the entire path was traced,
- model traffic was governed and measured,
- unsupported capabilities were documented instead of fabricated.

The project therefore represents a practical **Data + Retrieval + GenAI + LLMOps engineering workflow**, not just an LLM demo.
