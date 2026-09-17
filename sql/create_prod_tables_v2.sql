CREATE TABLE IF NOT EXISTS p14_rag_llmops.default.scientific_documents_prod (
  document_id STRING,
  abstract STRING,
  authors ARRAY<STRING>,
  created STRING,
  source STRING,
  source_url STRING,
  license STRING,
  ingested_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS p14_rag_llmops.default.scientific_chunks_prod (
  chunk_id STRING,
  document_id STRING,
  chunk_position INT,
  chunk_text STRING,
  token_count INT,
  chunking_strategy STRING,
  source STRING
) USING DELTA;

CREATE TABLE IF NOT EXISTS p14_rag_llmops.default.retrieval_eval_prod (
  query_id STRING,
  query_text STRING,
  relevant_document_ids ARRAY<STRING>,
  difficulty STRING,
  source STRING,
  created_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS p14_rag_llmops.default.retrieval_results_prod (
  run_id STRING,
  query_id STRING,
  retrieval_variant STRING,
  rank INT,
  chunk_id STRING,
  document_id STRING,
  score DOUBLE,
  latency_ms DOUBLE,
  created_at TIMESTAMP
) USING DELTA;
