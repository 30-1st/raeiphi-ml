"""
RAEIPHI — Parent-Child Document Chunker
=========================================
Splits the RAG knowledge base documents into two levels:

  Parent chunks: Full sections (~800–1500 tokens) — returned
  to the LLM as context when a child chunk matches a query.

  Child chunks: Small pieces (~200–300 tokens) — embedded and
  searched against for precise retrieval.

Each child stores a reference to its parent so the retrieval
system can return the full parent context.

Input:  ml/rag/documents/*.md
Output: ml/rag/chunks.json
"""

import json
import os
import re
import hashlib

# ── Config ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag")
DOCS_DIR = os.path.join(RAG_DIR, "documents")
CHUNKS_FILE = os.path.join(RAG_DIR, "chunks.json")

# Approximate token count (1 token ≈ 4 chars for English)
CHILD_TARGET_TOKENS = 250
CHILD_MAX_TOKENS = 400
CHILD_OVERLAP_SENTENCES = 1  # Overlap 1 sentence between child chunks


def estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters."""
    return len(text) // 4


def generate_id(text: str, prefix: str = "") -> str:
    """Generate a short deterministic ID from content."""
    h = hashlib.sha256(text.encode()).hexdigest()[:12]
    return f"{prefix}{h}"


def split_into_sections(markdown: str) -> list:
    """
    Split a markdown document into sections based on ## headers.
    Each section becomes a parent chunk.
    Returns list of (header, content) tuples.
    """
    # Split on ## headers (level 2)
    parts = re.split(r'\n(?=## )', markdown)
    sections = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract header if present
        lines = part.split('\n', 1)
        first_line = lines[0].strip()

        if first_line.startswith('## '):
            header = first_line.lstrip('#').strip()
            content = lines[1].strip() if len(lines) > 1 else ""
        elif first_line.startswith('# '):
            # Top-level header — this is the doc title, include as section
            header = first_line.lstrip('#').strip()
            content = lines[1].strip() if len(lines) > 1 else ""
        else:
            header = ""
            content = part

        if content:
            sections.append({
                "header": header,
                "content": content,
                "full_text": part,
            })

    return sections


def split_into_sentences(text: str) -> list:
    """Split text into sentences."""
    # Split on period, question mark, exclamation mark followed by space or newline
    # But preserve abbreviations and decimal numbers
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\"\*\-])', text)
    # Also split on double newlines (paragraph breaks)
    expanded = []
    for s in sentences:
        if '\n\n' in s:
            parts = s.split('\n\n')
            expanded.extend([p.strip() for p in parts if p.strip()])
        else:
            if s.strip():
                expanded.append(s.strip())
    return expanded


def create_child_chunks(parent_content: str, target_tokens: int = CHILD_TARGET_TOKENS) -> list:
    """
    Split parent content into smaller child chunks.
    Uses sentence boundaries to avoid cutting mid-sentence.
    Includes 1-sentence overlap between chunks for context continuity.
    """
    sentences = split_into_sentences(parent_content)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = estimate_tokens(sentence)

        # If adding this sentence exceeds target and we have content, save chunk
        if current_tokens + sentence_tokens > target_tokens and current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)

            # Start next chunk with overlap (last sentence of previous)
            if CHILD_OVERLAP_SENTENCES > 0 and current_chunk:
                overlap = current_chunk[-CHILD_OVERLAP_SENTENCES:]
                current_chunk = overlap
                current_tokens = sum(estimate_tokens(s) for s in overlap)
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        # Only add if it has meaningful content (not just overlap)
        if estimate_tokens(chunk_text) > 50:
            chunks.append(chunk_text)

    return chunks


def process_document(filepath: str) -> dict:
    """Process a single document into parent and child chunks."""
    filename = os.path.basename(filepath)
    doc_name = os.path.splitext(filename)[0]

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract document title (first # header)
    title_match = re.match(r'^# (.+)', content)
    doc_title = title_match.group(1).strip() if title_match else doc_name

    sections = split_into_sections(content)

    parents = []
    children = []

    for section in sections:
        # Create parent chunk
        parent_id = generate_id(section["full_text"], "p_")
        parent = {
            "id": parent_id,
            "type": "parent",
            "document": doc_name,
            "document_title": doc_title,
            "section_header": section["header"],
            "content": section["full_text"],
            "tokens_estimate": estimate_tokens(section["full_text"]),
        }
        parents.append(parent)

        # Create child chunks from this parent's content
        child_texts = create_child_chunks(section["content"])

        for i, child_text in enumerate(child_texts):
            child_id = generate_id(child_text + str(i), "c_")
            child = {
                "id": child_id,
                "type": "child",
                "parent_id": parent_id,
                "document": doc_name,
                "document_title": doc_title,
                "section_header": section["header"],
                "content": child_text,
                "chunk_index": i,
                "tokens_estimate": estimate_tokens(child_text),
            }
            children.append(child)

    return {
        "document": doc_name,
        "title": doc_title,
        "parents": parents,
        "children": children,
    }


def main():
    print("=" * 60)
    print("RAEIPHI — Parent-Child Document Chunker")
    print("=" * 60)

    # Find all markdown docs
    doc_files = sorted([
        os.path.join(DOCS_DIR, f)
        for f in os.listdir(DOCS_DIR)
        if f.endswith('.md')
    ])

    print(f"\nFound {len(doc_files)} documents:")
    for f in doc_files:
        print(f"  {os.path.basename(f)}")

    # Process all documents
    all_parents = []
    all_children = []
    doc_stats = []

    for filepath in doc_files:
        result = process_document(filepath)

        all_parents.extend(result["parents"])
        all_children.extend(result["children"])

        doc_stats.append({
            "document": result["document"],
            "title": result["title"],
            "parent_chunks": len(result["parents"]),
            "child_chunks": len(result["children"]),
        })

        print(f"\n  {result['document']}:")
        print(f"    Title:   {result['title']}")
        print(f"    Parents: {len(result['parents'])} sections")
        print(f"    Children:{len(result['children'])} chunks")

    # Save all chunks
    output = {
        "metadata": {
            "total_documents": len(doc_files),
            "total_parents": len(all_parents),
            "total_children": len(all_children),
            "child_target_tokens": CHILD_TARGET_TOKENS,
            "child_max_tokens": CHILD_MAX_TOKENS,
            "overlap_sentences": CHILD_OVERLAP_SENTENCES,
        },
        "documents": doc_stats,
        "parents": all_parents,
        "children": all_children,
    }

    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Summary
    parent_tokens = sum(p["tokens_estimate"] for p in all_parents)
    child_tokens = sum(c["tokens_estimate"] for c in all_children)

    print(f"\n{'=' * 60}")
    print("CHUNKING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Documents:      {len(doc_files)}")
    print(f"  Parent chunks:  {len(all_parents)} (~{parent_tokens:,} tokens)")
    print(f"  Child chunks:   {len(all_children)} (~{child_tokens:,} tokens)")
    print(f"  Avg child size: ~{child_tokens // max(len(all_children), 1)} tokens")
    print(f"  Output:         {CHUNKS_FILE}")
    print(f"  File size:      {os.path.getsize(CHUNKS_FILE) / 1024:.1f} KB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
