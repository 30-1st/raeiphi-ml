"""
RAEIPHI — RAG Retrieval System
================================
Takes a user query, embeds it locally using the same
sentence-transformers model used for indexing, searches
Supabase pgvector for matching child chunks, and returns
the full parent context for the top matches.

This module is imported by the AI chat layer. It can also
be run standalone to test retrieval quality.

Usage as module:
    from rag_retrieval import RAGRetriever
    retriever = RAGRetriever(supabase_url, supabase_key)
    results = retriever.retrieve("how many nodes do I need?")

Usage standalone (test mode):
    $env:SUPABASE_URL="https://lrbzbxvtawvevnwyseha.supabase.co"
    $env:SUPABASE_SERVICE_KEY="your-service-role-key"
    python src/rag_retrieval.py
"""

import json
import os
import sys
import time
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from supabase import create_client


# ── Config ──────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 5
DEFAULT_THRESHOLD = 0.3


class RAGRetriever:
    """
    Retrieves relevant RAEIPHI documentation context for a query
    using parent-child RAG with pgvector similarity search.

    Flow:
    1. Embed the query using the same model used for indexing
    2. Search rag_children for similar chunks (cosine similarity)
    3. Return the parent section content for each match
    4. Deduplicate parents (multiple children may share a parent)
    """

    def __init__(self, supabase_url: str, supabase_key: str):
        self.supabase = create_client(supabase_url, supabase_key)
        self.model = None  # Lazy load

    def _load_model(self):
        """Lazy-load the embedding model on first query."""
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_query(self, query: str) -> list:
        """Embed a user query into a vector."""
        self._load_model()
        embedding = self.model.encode(query)
        return embedding.tolist()

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> list:
        """
        Retrieve relevant parent context for a query.

        Returns a list of dicts:
        {
            "parent_content": str,    # Full parent section text
            "child_content": str,     # Matched child chunk
            "document_title": str,    # Source document
            "section_header": str,    # Section within document
            "similarity": float,      # Cosine similarity score
        }

        Results are deduplicated by parent — if multiple children
        from the same parent match, only the highest-scoring one
        is returned (but the parent content is the same full section).
        """
        start = time.time()

        # Step 1: Embed the query
        query_embedding = self.embed_query(query)
        embed_time = time.time() - start

        # Step 2: Search via Supabase RPC
        search_start = time.time()
        response = self.supabase.rpc(
            "match_rag_chunks",
            {
                "query_embedding": query_embedding,
                "match_threshold": threshold,
                "match_count": top_k * 2,  # Fetch extra to account for dedup
            },
        ).execute()
        search_time = time.time() - search_start

        if not response.data:
            return []

        # Step 3: Deduplicate by parent (keep highest similarity)
        seen_parents = {}
        for row in response.data:
            pid = row["parent_id"]
            if pid not in seen_parents or row["similarity"] > seen_parents[pid]["similarity"]:
                seen_parents[pid] = row

        # Step 4: Sort by similarity and limit to top_k
        results = sorted(
            seen_parents.values(),
            key=lambda x: x["similarity"],
            reverse=True,
        )[:top_k]

        # Format results
        formatted = []
        for r in results:
            formatted.append({
                "parent_content": r["parent_content"],
                "child_content": r["child_content"],
                "document_title": r["document_title"],
                "section_header": r["section_header"],
                "similarity": round(r["similarity"], 4),
            })

        total_time = time.time() - start
        self._last_timing = {
            "embed_ms": round(embed_time * 1000, 1),
            "search_ms": round(search_time * 1000, 1),
            "total_ms": round(total_time * 1000, 1),
        }

        return formatted

    def build_context(
        self,
        query: str,
        top_k: int = 3,
        threshold: float = DEFAULT_THRESHOLD,
        max_tokens: int = 3000,
    ) -> str:
        """
        Retrieve and format context for injection into an LLM prompt.

        Returns a single string of relevant parent sections,
        formatted with source headers, ready to paste into a
        system prompt or user message.

        Limits total context to approximately max_tokens.
        """
        results = self.retrieve(query, top_k=top_k, threshold=threshold)

        if not results:
            return ""

        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough token-to-char conversion

        for r in results:
            section = f"[Source: {r['document_title']} — {r['section_header']}]\n{r['parent_content']}"

            if total_chars + len(section) > max_chars:
                # Truncate this section to fit
                remaining = max_chars - total_chars
                if remaining > 200:
                    section = section[:remaining] + "\n[...truncated]"
                    context_parts.append(section)
                break

            context_parts.append(section)
            total_chars += len(section)

        return "\n\n---\n\n".join(context_parts)

    @property
    def last_timing(self) -> dict:
        """Timing breakdown of the last retrieve() call."""
        return getattr(self, "_last_timing", {})


# ── Standalone Test Mode ────────────────────────────────────────
def run_tests(retriever: RAGRetriever):
    """Run test queries to verify retrieval quality."""

    test_queries = [
        "How much does RAEIPHI cost?",
        "How many nodes do I need for a two-bedroom apartment?",
        "Does RAEIPHI record video or audio?",
        "How does WiFi sensing detect people?",
        "Can RAEIPHI detect water leaks?",
        "What is the monthly subscription price?",
        "How do I install the nodes?",
        "Can I use RAEIPHI in a hotel?",
        "What happens if WiFi goes down?",
        "Does it work for staffless hotels?",
    ]

    print(f"\n{'=' * 60}")
    print("RAEIPHI — RAG Retrieval Tests")
    print(f"{'=' * 60}")
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Test queries: {len(test_queries)}")

    all_results = []

    for i, query in enumerate(test_queries):
        print(f"\n{'─' * 60}")
        print(f"  Query {i + 1}: \"{query}\"")
        print(f"{'─' * 60}")

        results = retriever.retrieve(query, top_k=3)
        timing = retriever.last_timing

        if not results:
            print("  No results found.")
            all_results.append({"query": query, "results": 0})
            continue

        all_results.append({
            "query": query,
            "results": len(results),
            "top_similarity": results[0]["similarity"],
            "timing": timing,
        })

        for j, r in enumerate(results):
            print(f"\n  Result {j + 1} (similarity: {r['similarity']:.4f}):")
            print(f"    Source: {r['document_title']} — {r['section_header']}")
            print(f"    Child:  {r['child_content'][:120]}...")

        print(f"\n  Timing: embed={timing['embed_ms']}ms, "
              f"search={timing['search_ms']}ms, "
              f"total={timing['total_ms']}ms")

    # Summary
    print(f"\n{'=' * 60}")
    print("TEST SUMMARY")
    print(f"{'=' * 60}")

    queries_with_results = [r for r in all_results if r["results"] > 0]
    if queries_with_results:
        avg_sim = sum(r["top_similarity"] for r in queries_with_results) / len(queries_with_results)
        avg_time = sum(r["timing"]["total_ms"] for r in queries_with_results) / len(queries_with_results)
        print(f"  Queries with results: {len(queries_with_results)}/{len(test_queries)}")
        print(f"  Avg top similarity:   {avg_sim:.4f}")
        print(f"  Avg retrieval time:   {avg_time:.0f}ms")
    else:
        print("  No queries returned results — check embeddings.")

    # Test build_context
    print(f"\n{'─' * 60}")
    print("  build_context() test:")
    print(f"{'─' * 60}")
    context = retriever.build_context("How much does it cost and how do I set it up?")
    print(f"  Context length: {len(context)} chars (~{len(context)//4} tokens)")
    print(f"  Preview:\n{context[:500]}...")

    print(f"\n{'=' * 60}")
    print("RETRIEVAL TESTS COMPLETE")
    print(f"{'=' * 60}")


def main():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.")
        print('  $env:SUPABASE_URL="https://lrbzbxvtawvevnwyseha.supabase.co"')
        print('  $env:SUPABASE_SERVICE_KEY="your-service-role-key"')
        sys.exit(1)

    retriever = RAGRetriever(supabase_url, supabase_key)
    run_tests(retriever)


if __name__ == "__main__":
    main()
