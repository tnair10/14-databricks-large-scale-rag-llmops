CREATE TABLE IF NOT EXISTS p14_rag_llmops.default.scientific_documents_prod (
  document_id STRING,
  title STRING,
  abstract STRING,
  categories ARRAY<STRING>,
  published_date STRING,
  source STRING,
  source_url STRING,
  ingested_at TIMESTAMP
) USING DELTA;

CREATE TABLE IF NOT EXISTS p14_rag_llmops.default.scientific_chunks_prod (
  chunk_id STRING,
  document_id STRING,
  title STRING,
  categories ARRAY<STRING>,
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
