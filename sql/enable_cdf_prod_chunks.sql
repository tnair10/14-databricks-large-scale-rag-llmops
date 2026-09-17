ALTER TABLE p14_rag_llmops.default.scientific_chunks_prod
SET TBLPROPERTIES (delta.enableChangeDataFeed = true);

SHOW TBLPROPERTIES p14_rag_llmops.default.scientific_chunks_prod;
