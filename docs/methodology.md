# Project 14 Methodology

## Objective
Build and empirically evaluate a Databricks-native large-scale scientific RAG platform.

## Scale stages
- Development: 10,000 documents
- Validation: 100,000 documents
- Scale: 500,000 documents
- Final: 1,000,000+ documents

## Retrieval variants
1. Lexical baseline
2. Vector retrieval
3. Hybrid retrieval
4. Hybrid + reranker
5. Hybrid + reranker + optimized context

## Retrieval metrics
Recall@5, Recall@10, MRR, NDCG@10, P50 latency, P95 latency.

## GenAI metrics
Answer correctness, groundedness, context precision, context recall,
unsupported-claim rate, token usage, latency, and cost per query.

## Evidence policy
Synthetic development results validate implementation only.
Portfolio conclusions will use real public scientific data and actual Databricks measurements.
No estimated value is presented as an observed result.
