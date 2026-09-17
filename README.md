# Databricks Large-Scale Scientific RAG & LLMOps Benchmark Platform

A Databricks-native retrieval-augmented generation and LLMOps project that builds, evaluates, and traces scientific RAG workflows over a real arXiv-derived corpus.

## What this project demonstrates

- Unity Catalog / Delta Lake ingestion for scientific documents and chunks
- Databricks AI Search with managed embeddings
- ANN and HYBRID retrieval benchmarking
- Real-corpus retrieval evaluation with 500 deterministic queries
- Unity Gateway model service for grounded generation
- MLflow experiment tracking and end-to-end GenAI tracing
- Request/token usage telemetry and governance controls
- Development-to-real-data validation workflow

## Architecture

```text
arXiv-derived scientific corpus
        |
        v
Unity Catalog Volume
        |
        v
Delta documents + chunks
        |
        v
Databricks AI Search
   ANN / HYBRID
        |
        v
Top-k retrieved chunks
        |
        v
Grounded prompt builder
        |
        v
Unity Gateway model service
        |
        v
Grounded answer
        |
        v
MLflow tracing + metrics
```

## Real-data corpus

The validated real-data baseline contains:

- 10,000 scientific documents
- 11,528 chunks
- 0 empty chunks
- 0 documents without chunks
- 0 duplicate chunk IDs
- average 110.35 tokens per chunk
- P50 104 tokens
- P95 192 tokens

The real AI Search index contains all **11,528** chunks.

## Real retrieval evaluation

A deterministic 500-query evaluation set was generated from real arXiv abstracts with one known relevant document per query.

### Corrected results

| Metric | ANN | HYBRID |
|---|---:|---:|
| Hit@5 | 0.984 | **0.996** |
| Hit@10 | 0.990 | **0.998** |
| MRR | 0.9568 | **0.9821** |
| NDCG@10 | 0.9651 | **0.9860** |
| P50 latency | **469.81 ms** | 479.31 ms |
| P95 latency | 630.37 ms | **615.64 ms** |
| Failures | 0 | 0 |

HYBRID improved retrieval quality in this measured run while keeping latency close to ANN.

> The query set is deterministic and derived from corpus text. It is useful for repeatable retrieval benchmarking but is not presented as a human-authored gold-standard relevance dataset.

## End-to-end real RAG

The final real-data RAG path uses:

```text
Question
 -> HYBRID AI Search
 -> top-5 real arXiv chunks
 -> grounded prompt
 -> Unity Gateway model service
 -> final answer
 -> MLflow trace
```

Final measured run:

- end-to-end latency: **2.55 s**
- input tokens: **1,219**
- output tokens: **305**
- total tokens: **1,524**
- status: `RAG_REAL_E2E_FINAL_ONLY=PASS`

The final implementation filters reasoning-model internals and returns only assistant `output_text`.

## LLMOps and governance

The project uses:

- MLflow experiments for retrieval metrics
- MLflow tracing for retrieval / prompt / generation spans
- Unity Gateway model service
- Unity Gateway usage tracking
- inference-table configuration
- service-level request/token rate limiting

## Reranker experiment

Managed reranking was tested as an intended retrieval variant. The workspace exposed the reranker feature, but API requests returned a workspace-level configuration restriction preventing access to the reranker model. ANN and HYBRID evaluation therefore completed successfully while managed reranking is documented as unavailable in this environment.

## Development plumbing versus final evidence

A 28K synthetic development corpus was used first to validate:

- chunking
- AI Search configuration
- ANN/HYBRID API usage
- MLflow integration
- RAG tracing

Final benchmark claims in this README use the **real 10K-document arXiv-derived corpus**, not the synthetic development corpus.

## Evidence

UI evidence is available under:

```text
docs/evidence/ui/
```

Key screenshots include:

- `ai_search_real_index_online_11528.png`
- `ai_search_real_initial_sync_completed.png`
- `mlflow_real_ai_search_500_corrected_ann_metrics.png`
- `mlflow_real_ai_search_500_corrected_quality_metrics.png`
- `mlflow_real_ai_search_500_corrected_hybrid_metrics.png`
- `mlflow_real_rag_end_to_end_trace.png`
- `unity_gateway_usage_tracking_rag_requests.png`
- `ai_playground_grounded_rag_example.png`

## Important measurement notes

- All performance values above are measurements from this specific Databricks Free Edition workspace and workload.
- They are not universal Databricks performance claims.
- Managed embedding throughput varied during initial sync and stabilized after startup.
- Early NDCG calculations that allowed duplicate chunks from one relevant document to create gain above 1 were corrected; only the corrected metrics are used here.

## Security

Credentials are expected through `.env` and must never be committed.

Expected variables include:

```text
DATABRICKS_SERVER_HOSTNAME
DATABRICKS_HTTP_PATH
DATABRICKS_TOKEN
DATABRICKS_CATALOG
DATABRICKS_SCHEMA
```

The `.env` file should remain gitignored.
