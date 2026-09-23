
-- RAEIPHI RAG Setup (local embeddings, 384 dimensions)
-- Run this in Supabase SQL Editor. If tables already exist from
-- the previous version, this drops and recreates them.

DROP TABLE IF EXISTS rag_children CASCADE;
DROP TABLE IF EXISTS rag_parents CASCADE;

CREATE TABLE rag_parents (
    id TEXT PRIMARY KEY,
    document TEXT NOT NULL,
    document_title TEXT NOT NULL,
    section_header TEXT,
    content TEXT NOT NULL,
    tokens_estimate INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE rag_children (
    id TEXT PRIMARY KEY,
    parent_id TEXT NOT NULL REFERENCES rag_parents(id) ON DELETE CASCADE,
    document TEXT NOT NULL,
    document_title TEXT NOT NULL,
    section_header TEXT,
    content TEXT NOT NULL,
    chunk_index INTEGER,
    tokens_estimate INTEGER,
    embedding VECTOR(384),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_rag_children_embedding
ON rag_children USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 10);

CREATE INDEX idx_rag_children_parent_id
ON rag_children(parent_id);

CREATE OR REPLACE FUNCTION match_rag_chunks(
    query_embedding VECTOR(384),
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    child_id TEXT,
    child_content TEXT,
    parent_id TEXT,
    parent_content TEXT,
    document TEXT,
    document_title TEXT,
    section_header TEXT,
    similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id AS child_id,
        c.content AS child_content,
        p.id AS parent_id,
        p.content AS parent_content,
        c.document,
        c.document_title,
        c.section_header,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM rag_children c
    JOIN rag_parents p ON c.parent_id = p.id
    WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

ALTER TABLE rag_parents ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_children ENABLE ROW LEVEL SECURITY;

DO $$
BEGIN
    DROP POLICY IF EXISTS "Allow public read on rag_parents" ON rag_parents;
    DROP POLICY IF EXISTS "Allow public read on rag_children" ON rag_children;
END $$;

CREATE POLICY "Allow public read on rag_parents"
    ON rag_parents FOR SELECT
    USING (true);

CREATE POLICY "Allow public read on rag_children"
    ON rag_children FOR SELECT
    USING (true);
