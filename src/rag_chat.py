"""
RAEIPHI — RAG-Powered AI Chat
================================
Combines the RAG retrieval system with Groq LLM to create
a grounded AI chat that answers from RAEIPHI documentation.

Flow:
  1. User asks a question
  2. RAG retriever finds relevant documentation
  3. Documentation context is injected into the LLM prompt
  4. Groq generates a response grounded in real product data
  5. Response includes source attribution

Usage as module:
    from rag_chat import RAGChat
    chat = RAGChat(supabase_url, supabase_key, groq_key)
    response = chat.ask("How much does RAEIPHI cost?")

Usage standalone (interactive chat):
    $env:SUPABASE_URL="https://lrbzbxvtawvevnwyseha.supabase.co"
    $env:SUPABASE_SERVICE_KEY="your-service-role-key"
    $env:GROQ_API_KEY="your-groq-key"
    python src/rag_chat.py
"""

import os
import sys
import json
import time
from typing import Optional

from groq import Groq
from rag_retrieval import RAGRetriever


# ── Config ──────────────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-120b"
MAX_CONTEXT_TOKENS = 3000
MAX_RESPONSE_TOKENS = 1024
TEMPERATURE = 0.3  # Low temp for factual, grounded responses

SYSTEM_PROMPT = """You are RAEIPHI's AI assistant — a knowledgeable, helpful, and direct product expert for RAEIPHI, a WiFi-based occupancy sensing platform.

Your role:
- Answer customer questions about RAEIPHI's products, pricing, technology, setup, and capabilities
- Help prospective customers understand whether RAEIPHI fits their needs
- Provide accurate pricing and node recommendations based on property details
- Explain the WiFi CSI sensing technology in accessible terms
- For staffless hospitality, co-working spaces, and short-term rental hosts, explain the specific value RAEIPHI provides

Rules:
- ONLY answer from the documentation context provided below. Do not invent features, prices, or capabilities not in the context.
- If the context doesn't contain enough information to answer, say so honestly and suggest the customer contact RAEIPHI directly.
- When discussing pricing, always mention the $19.99/month SaaS subscription alongside hardware costs.
- Always recommend a minimum of 4 nodes per property for viable accuracy.
- Be conversational and helpful, not robotic. You're a product expert, not a search engine.
- Keep responses concise — 2-4 paragraphs max unless the question requires more detail.
- When relevant, mention which document section your answer draws from.

Current pricing (always use these exact numbers):
- DIY Node: $49 per node + shipping and handling
- Full Package: $89 per package + shipping and handling
- SaaS Subscription: $19.99/month (covers all nodes and properties)
- Minimum: 4 nodes per property for viable accuracy"""

CONTEXT_TEMPLATE = """
--- RETRIEVED DOCUMENTATION ---
The following sections from RAEIPHI's knowledge base are relevant to the user's question. Use this information to answer accurately.

{context}

--- END DOCUMENTATION ---
"""


class RAGChat:
    """
    RAG-powered chat that retrieves documentation context
    and generates grounded responses via Groq.
    """

    def __init__(self, supabase_url: str, supabase_key: str, groq_api_key: str):
        self.retriever = RAGRetriever(supabase_url, supabase_key)
        self.groq = Groq(api_key=groq_api_key)
        self.conversation_history = []

    def ask(
        self,
        question: str,
        top_k: int = 3,
        include_sources: bool = True,
    ) -> dict:
        """
        Ask a question and get a RAG-grounded response.

        Returns:
        {
            "answer": str,           # The generated response
            "sources": list,         # Source documents used
            "retrieval_time_ms": float,
            "generation_time_ms": float,
            "total_time_ms": float,
        }
        """
        total_start = time.time()

        # Step 1: Retrieve relevant context
        retrieval_start = time.time()
        context = self.retriever.build_context(
            question,
            top_k=top_k,
            max_tokens=MAX_CONTEXT_TOKENS,
        )
        retrieval_results = self.retriever.retrieve(question, top_k=top_k)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # Step 2: Build the prompt
        if context:
            context_block = CONTEXT_TEMPLATE.format(context=context)
        else:
            context_block = "\n[No relevant documentation found for this query.]\n"

        # Build messages with conversation history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + context_block},
        ]

        # Add conversation history (keep last 6 turns for context)
        for msg in self.conversation_history[-6:]:
            messages.append(msg)

        # Add current question
        messages.append({"role": "user", "content": question})

        # Step 3: Generate response via Groq
        gen_start = time.time()
        completion = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_RESPONSE_TOKENS,
        )
        generation_time = (time.time() - gen_start) * 1000

        answer = completion.choices[0].message.content

        # Step 4: Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Step 5: Compile sources
        sources = []
        for r in retrieval_results:
            sources.append({
                "document": r["document_title"],
                "section": r["section_header"],
                "similarity": r["similarity"],
            })

        total_time = (time.time() - total_start) * 1000

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_time_ms": round(retrieval_time, 1),
            "generation_time_ms": round(generation_time, 1),
            "total_time_ms": round(total_time, 1),
            "context_used": bool(context),
            "model": GROQ_MODEL,
        }

    def reset_conversation(self):
        """Clear conversation history."""
        self.conversation_history = []


# ── Interactive Chat Mode ───────────────────────────────────────
def interactive_chat(chat: RAGChat):
    """Run an interactive chat session in the terminal."""
    print("=" * 60)
    print("RAEIPHI — RAG-Powered AI Chat")
    print("=" * 60)
    print(f"  Model:     {GROQ_MODEL} (via Groq)")
    print(f"  Retrieval: parent-child RAG with pgvector")
    print(f"  Context:   up to {MAX_CONTEXT_TOKENS} tokens per query")
    print()
    print("  Type your question and press Enter.")
    print("  Type 'quit' to exit, 'reset' to clear history,")
    print("  'sources' to toggle source display.")
    print("=" * 60)

    show_sources = True
    show_timing = True

    while True:
        print()
        try:
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not question:
            continue

        if question.lower() == "quit":
            print("  Goodbye!")
            break

        if question.lower() == "reset":
            chat.reset_conversation()
            print("  Conversation history cleared.")
            continue

        if question.lower() == "sources":
            show_sources = not show_sources
            print(f"  Source display: {'ON' if show_sources else 'OFF'}")
            continue

        if question.lower() == "timing":
            show_timing = not show_timing
            print(f"  Timing display: {'ON' if show_timing else 'OFF'}")
            continue

        # Get response
        try:
            result = chat.ask(question, include_sources=show_sources)
        except Exception as e:
            print(f"\n  Error: {e}")
            continue

        # Display answer
        print(f"\n  RAEIPHI: {result['answer']}")

        # Display sources
        if show_sources and result["sources"]:
            print(f"\n  Sources:")
            for s in result["sources"]:
                print(f"    - {s['document']} > {s['section']} "
                      f"(similarity: {s['similarity']:.3f})")

        # Display timing
        if show_timing:
            print(f"\n  Timing: retrieval={result['retrieval_time_ms']:.0f}ms, "
                  f"generation={result['generation_time_ms']:.0f}ms, "
                  f"total={result['total_time_ms']:.0f}ms")


def run_demo(chat: RAGChat):
    """Run a scripted demo with sample questions."""
    demo_questions = [
        "How much does RAEIPHI cost for a two-bedroom Airbnb?",
        "Does the system record any audio or video?",
        "How does it work for staffless hotels?",
        "Can it detect water leaks?",
        "What if my WiFi goes down?",
    ]

    print("=" * 60)
    print("RAEIPHI — RAG Chat Demo")
    print("=" * 60)

    for i, question in enumerate(demo_questions):
        print(f"\n{'─' * 60}")
        print(f"  Demo Question {i + 1}: \"{question}\"")
        print(f"{'─' * 60}")

        result = chat.ask(question)

        print(f"\n  Answer: {result['answer']}")
        print(f"\n  Sources: {', '.join(s['document'] + ' > ' + s['section'] for s in result['sources'][:2])}")
        print(f"  Timing: {result['total_time_ms']:.0f}ms total")

        # Reset between demo questions to avoid context bleed
        chat.reset_conversation()

    print(f"\n{'=' * 60}")
    print("DEMO COMPLETE")
    print(f"{'=' * 60}")


def main():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    if not supabase_url or not supabase_key:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.")
        sys.exit(1)

    if not groq_key:
        print("ERROR: Set GROQ_API_KEY env var.")
        print('  $env:GROQ_API_KEY="gsk_your-groq-key"')
        sys.exit(1)

    chat = RAGChat(supabase_url, supabase_key, groq_key)

    # Check for demo flag
    if "--demo" in sys.argv:
        run_demo(chat)
    else:
        interactive_chat(chat)


if __name__ == "__main__":
    main()
