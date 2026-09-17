# Development validation

- Unity Catalog catalog: `p14_rag_llmops`
- Development documents: 10,000
- Development chunks: 28,000
- Categories: 8
- Server-side Delta generation/load: 6.44 s
- Chunk validation: PASS
- Empty chunks: 0
- Oversized chunks: 0
- Documents without chunks: 0

## Lexical plumbing baseline
- Evaluation queries: 500
- Recall@5: 1.0
- Recall@10: 1.0
- MRR: 1.0
- NDCG@10: 1.0
- P50: 728.92 ms
- P95: 1047.51 ms

These quality scores are not production retrieval evidence. The synthetic
queries expose their category phrase and validate the metric pipeline only.

## AI Search development index
- Endpoint: `p14-ai-search-dev`
- Index: `scientific_chunks_dev_ai_index`
- Type: Delta Sync / Hybrid
- Embedding model: `databricks-qwen3-embedding-0-6b`
- Update mode: Triggered
- Initial observed managed-embedding throughput: ~4 rows/s
