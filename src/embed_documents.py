"""
RAEIPHI — Embedding Pipeline (Local)
======================================
Generates embeddings using sentence-transformers (free, local,
no API key needed) and uploads to Supabase pgvector.

Prerequisites:
  pip install sentence-transformers supabase

Usage:
  First update the Supabase tables (dimension changed from 1536 to 384):
    python src/embed_documents.py --setup-sql

  Then generate and upload:
    $env:SUPABASE_URL="https://lrbzbxvtawvevnwyseha.supabase.co"
    $env:SUPABASE_SERVICE_KEY="your-service-role-key"
    python src/embed_documents.py
"""

import json
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag")
CHUNKS_FILE = os.path.join(RAG_DIR, "chunks.json")
EMBEDDINGS_CACHE = os.path.join(RAG_DIR, "embeddings_cache.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384
BATCH_SIZE = 50


SETUP_SQL = """
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
"""


def print_setup_sql():
    print("=" * 60)
    print("RAEIPHI — Supabase SQL Setup for RAG (384-dim)")
    print("=" * 60)
    print("\nCopy and paste the following SQL into your Supabase SQL Editor:\n")
    print(SETUP_SQL)
    print("\n" + "=" * 60)
    print("After running the SQL successfully, run this script again")
    print("without the --setup-sql flag to generate and upload embeddings.")
    print("=" * 60)
    sql_file = os.path.join(RAG_DIR, "setup_supabase.sql")
    with open(sql_file, 'w') as f:
        f.write(SETUP_SQL)
    print(f"\nSQL also saved to: {sql_file}")


def load_chunks():
    if not os.path.exists(CHUNKS_FILE):
        print(f"ERROR: {CHUNKS_FILE} not found. Run chunk_documents.py first.")
        sys.exit(1)
    with open(CHUNKS_FILE, 'r') as f:
        return json.load(f)


def generate_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    print(f"  Loading model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Encoding {len(texts)} texts...")
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=BATCH_SIZE)
    elapsed = time.time() - start
    embeddings_list = [emb.tolist() for emb in embeddings]
    print(f"  Generated {len(embeddings_list)} embeddings in {elapsed:.1f}s")
    print(f"  Dimensions: {len(embeddings_list[0])}")
    return embeddings_list


def upload_to_supabase(parents, children, embeddings, supabase_url, supabase_key):
    from supabase import create_client
    supabase = create_client(supabase_url, supabase_key)

    print("\n  Clearing existing RAG data...")
    try:
        supabase.table("rag_children").delete().neq("id", "").execute()
        supabase.table("rag_parents").delete().neq("id", "").execute()
    except Exception as e:
        print(f"  Note: {e}")
        print("  Tables may be empty, continuing...")

    print(f"  Uploading {len(parents)} parent chunks...")
    parent_records = [{
        "id": p["id"], "document": p["document"],
        "document_title": p["document_title"],
        "section_header": p.get("section_header", ""),
        "content": p["content"],
        "tokens_estimate": p.get("tokens_estimate", 0),
    } for p in parents]

    for i in range(0, len(parent_records), 50):
        supabase.table("rag_parents").insert(parent_records[i:i+50]).execute()
    print(f"    {len(parent_records)} parents uploaded")

    print(f"  Uploading {len(children)} child chunks with embeddings...")
    for i, (child, embedding) in enumerate(zip(children, embeddings)):
        record = {
            "id": child["id"], "parent_id": child["parent_id"],
            "document": child["document"],
            "document_title": child["document_title"],
            "section_header": child.get("section_header", ""),
            "content": child["content"],
            "chunk_index": child.get("chunk_index", 0),
            "tokens_estimate": child.get("tokens_estimate", 0),
            "embedding": embedding,
        }
        supabase.table("rag_children").insert(record).execute()
        if (i + 1) % 25 == 0 or i == len(children) - 1:
            print(f"    {i + 1}/{len(children)} uploaded...")
    print("  Upload complete.")


def save_cache(children, embeddings):
    cache = {
        "model": EMBEDDING_MODEL, "dimensions": EMBEDDING_DIMENSIONS,
        "count": len(embeddings),
        "items": [{"id": c["id"], "parent_id": c["parent_id"],
                   "content_preview": c["content"][:100], "embedding": e}
                  for c, e in zip(children, embeddings)],
    }
    with open(EMBEDDINGS_CACHE, 'w') as f:
        json.dump(cache, f)
    size_mb = os.path.getsize(EMBEDDINGS_CACHE) / 1024 / 1024
    print(f"  Cache saved: {EMBEDDINGS_CACHE} ({size_mb:.1f} MB)")


def main():
    if "--setup-sql" in sys.argv:
        print_setup_sql()
        return

    print("=" * 60)
    print("RAEIPHI — Embedding Pipeline (Local)")
    print("=" * 60)
    print(f"  Model: {EMBEDDING_MODEL} (free, runs locally)")
    print(f"  Dimensions: {EMBEDDING_DIMENSIONS}")

    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not supabase_url or not supabase_key:
        print("\nERROR: Supabase environment variables not set.")
        print('  $env:SUPABASE_URL="https://lrbzbxvtawvevnwyseha.supabase.co"')
        print('  $env:SUPABASE_SERVICE_KEY="your-service-role-key"')
        sys.exit(1)
    print(f"  Supabase URL: {supabase_url[:40]}...")

    print("\nLoading chunks...")
    data = load_chunks()
    parents, children = data["parents"], data["children"]
    print(f"  Parent chunks: {len(parents)}")
    print(f"  Child chunks:  {len(children)}")

    print(f"\nGenerating embeddings for {len(children)} child chunks...")
    child_texts = [c["content"] for c in children]
    embeddings = generate_embeddings(child_texts)

    print("\nSaving local cache...")
    save_cache(children, embeddings)

    print("\nUploading to Supabase...")
    upload_to_supabase(parents, children, embeddings, supabase_url, supabase_key)

    print(f"\n{'=' * 60}")
    print("EMBEDDING PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Child chunks embedded: {len(children)}")
    print(f"  Parent chunks stored:  {len(parents)}")
    print(f"  Embedding model:       {EMBEDDING_MODEL}")
    print(f"  Vector dimensions:     {EMBEDDING_DIMENSIONS}")
    print(f"  API cost:              $0.00 (local model)")
    print(f"  Supabase tables:       rag_parents, rag_children")
    print(f"  Search function:       match_rag_chunks()")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()