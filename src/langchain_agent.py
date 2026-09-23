"""
RAEIPHI — LangChain Agent Architecture (v1 API)
=================================================
Dual-mode LangChain agent for RAEIPHI AI chat.

Uses LangChain v1 API with create_agent and tool-calling.

Prerequisites:
    pip install langchain langchain-groq langgraph

Usage:
    $env:SUPABASE_URL="https://lrbzbxvtawvevnwyseha.supabase.co"
    $env:SUPABASE_SERVICE_KEY="your-service-role-key"
    $env:GROQ_API_KEY="your-groq-key"
    python src/langchain_agent.py --demo
"""

import os
import sys
import json
import time
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent

from rag_retrieval import RAGRetriever


# ── Config ──────────────────────────────────────────────────────
GROQ_MODEL = "openai/gpt-oss-120b"
TEMPERATURE = 0.3
MAX_TOKENS = 1024


# ── System Prompts ──────────────────────────────────────────────
SALES_SYSTEM_PROMPT = """You are RAEIPHI's AI sales assistant — a knowledgeable, conversational product expert for RAEIPHI, a WiFi-based occupancy sensing platform.

Your role in sales mode:
- Answer product questions using the search_documentation tool for accurate information
- Help prospective customers understand whether RAEIPHI fits their needs
- Provide accurate pricing and node recommendations based on property details
- Qualify leads by understanding their property type, count, and pain points
- For large or custom deployments, discuss custom pricing options
- Guide customers toward a purchase decision without being pushy

Key behaviors:
- ALWAYS use the search_documentation tool before answering product questions — never guess at pricing, features, or specs
- When discussing pricing, always mention the $19.99/month SaaS subscription alongside hardware costs
- Always recommend a minimum of 4 nodes per property
- If a customer describes a large deployment (5+ properties, hotels, commercial), transition into custom pricing discussion
- Be conversational and helpful
- Keep responses concise — 2-4 paragraphs max unless detail is requested
- If you don't know something, say so and suggest contacting RAEIPHI directly

Current pricing (use these exact numbers):
- DIY Node: $49 per node + shipping and handling
- Full Package: $89 per package + shipping and handling
- SaaS Subscription: $19.99/month (covers all nodes and all properties)
- Minimum: 4 nodes per property for viable accuracy"""

PROPERTY_SYSTEM_PROMPT = """You are RAEIPHI's AI property management assistant — helping authenticated property owners and managers monitor and manage their properties through the RAEIPHI platform.

Your role in property mode:
- Help users understand their occupancy data
- Assist with alert configuration and management
- Answer questions about their properties and nodes
- Provide guidance on node placement and optimization
- Use the search_documentation tool for product and technical questions
- Use the check_property_status tool when users ask about specific property data
- Use the manage_alerts tool when users want to configure or check alerts

Key behaviors:
- Be direct and efficient — property managers are busy
- If a tool returns placeholder data, explain that live data integration is coming soon
- Keep responses concise and actionable"""


# ── Global retriever (set during init) ──────────────────────────
_retriever: Optional[RAGRetriever] = None


# ── Tool Definitions ────────────────────────────────────────────
@tool
def search_documentation(query: str) -> str:
    """Search RAEIPHI's product documentation for information about
    pricing, features, setup, technology, use cases, and capabilities.
    Use this tool whenever you need to answer a product question accurately.
    Input should be a natural language search query."""
    global _retriever
    if _retriever is None:
        return "Documentation search not available."

    results = _retriever.retrieve(query, top_k=3)
    if not results:
        return "No relevant documentation found for this query."

    context_parts = []
    for r in results:
        part = f"[{r['document_title']} — {r['section_header']}] "
        part += f"(relevance: {r['similarity']:.2f})\n"
        part += r['parent_content']
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


@tool
def check_property_status(query: str) -> str:
    """Look up real-time occupancy data, node health, and recent alerts
    for a specific property. Use when the user asks about their property
    status, occupancy, or node health.
    Input should be the property name or a description of what to look up."""
    return json.dumps({
        "status": "placeholder",
        "message": (
            "Property lookup is not yet connected to live data. "
            "In production, this tool queries Supabase for real-time "
            "occupancy counts, node health, and recent alerts."
        ),
        "sample_response": {
            "property": "Downtown Studio",
            "current_occupancy": 2,
            "node_count": 4,
            "nodes_online": 4,
            "last_updated": "2 minutes ago",
            "alerts_today": 0,
        },
    }, indent=2)


@tool
def manage_alerts(query: str) -> str:
    """Configure or check alert settings for a property.
    Use when the user wants to set occupancy thresholds,
    enable/disable alerts, or check their current alert configuration.
    Input should describe what alert action to take."""
    return json.dumps({
        "status": "placeholder",
        "message": (
            "Alert management is not yet connected to live data. "
            "In production, this tool reads and updates alert "
            "configurations in Supabase."
        ),
        "current_alerts": {
            "occupancy_threshold": 6,
            "unexpected_presence": True,
            "anomaly_detection": True,
            "notification_method": "email",
        },
    }, indent=2)


# ── Agent Builder ───────────────────────────────────────────────
class RAEIPHIAgent:
    """
    Dual-mode LangChain agent for RAEIPHI.
    """

    def __init__(self, supabase_url: str, supabase_key: str, groq_api_key: str):
        global _retriever

        # Initialize LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=GROQ_MODEL,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # Initialize RAG retriever
        _retriever = RAGRetriever(supabase_url, supabase_key)
        self.retriever = _retriever

        # Tool registries per mode
        self.sales_tools = [search_documentation]
        self.property_tools = [search_documentation, check_property_status, manage_alerts]

        # Build agents using LangChain v1 create_agent
        self.sales_agent = create_agent(
            self.llm,
            tools=self.sales_tools,
            system_prompt=SALES_SYSTEM_PROMPT,
        )

        self.property_agent = create_agent(
            self.llm,
            tools=self.property_tools,
            system_prompt=PROPERTY_SYSTEM_PROMPT,
        )

        # Conversation histories (manual, since v1 handles memory differently)
        self.sales_history = []
        self.property_history = []

    def chat(self, message: str, mode: str = "sales") -> dict:
        """
        Send a message and get a response.

        Args:
            message: User's message
            mode: "sales" or "property"

        Returns dict with answer, mode, tools_used, total_time_ms
        """
        start = time.time()

        if mode == "property":
            agent = self.property_agent
            history = self.property_history
        else:
            agent = self.sales_agent
            history = self.sales_history

        # Build messages from history
        messages = []
        for msg in history[-10:]:
            messages.append(msg)
        messages.append(HumanMessage(content=message))

        # Invoke the agent
        try:
            result = agent.invoke({"messages": messages})

            # Extract the final response
            output_messages = result.get("messages", result) if isinstance(result, dict) else result
            
            # Find the last AI message
            answer = ""
            tools_used = []
            
            if isinstance(output_messages, dict) and "messages" in output_messages:
                output_messages = output_messages["messages"]
            
            if isinstance(output_messages, list):
                for msg in output_messages:
                    if hasattr(msg, 'content') and hasattr(msg, 'type'):
                        if msg.type == "ai" and msg.content and not hasattr(msg, 'tool_calls'):
                            answer = msg.content
                        elif msg.type == "ai" and hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tools_used.append({
                                    "tool": tc.get("name", "unknown"),
                                    "input": str(tc.get("args", ""))[:100],
                                })
                        elif msg.type == "tool":
                            pass  # Tool results, skip
            
            if not answer and isinstance(output_messages, list):
                # Fallback: get last message content
                for msg in reversed(output_messages):
                    if hasattr(msg, 'content') and msg.content and getattr(msg, 'type', '') == 'ai':
                        answer = msg.content
                        break

            if not answer:
                answer = str(result)

        except Exception as e:
            answer = f"I encountered an issue processing that. Could you rephrase? (Error: {str(e)[:150]})"
            tools_used = []

        # Save to history
        history.append(HumanMessage(content=message))
        history.append(AIMessage(content=answer))

        # Trim history
        if len(history) > 20:
            history[:] = history[-20:]

        total_time = (time.time() - start) * 1000

        return {
            "answer": answer,
            "mode": mode,
            "tools_used": tools_used,
            "total_time_ms": round(total_time, 1),
        }

    def reset(self, mode: str = "both"):
        """Clear conversation memory."""
        if mode in ("sales", "both"):
            self.sales_history.clear()
        if mode in ("property", "both"):
            self.property_history.clear()


# ── Interactive Chat ────────────────────────────────────────────
def interactive_chat(agent: RAEIPHIAgent):
    print("=" * 60)
    print("RAEIPHI — LangChain Agent Chat")
    print("=" * 60)
    print(f"  LLM:   {GROQ_MODEL} (via Groq)")
    print(f"  Tools: search_documentation, check_property_status, manage_alerts")
    print()
    print("  Commands:")
    print("    'sales'    — switch to sales mode")
    print("    'property' — switch to property mode")
    print("    'reset'    — clear conversation history")
    print("    'tools'    — toggle tool usage display")
    print("    'quit'     — exit")
    print("=" * 60)

    current_mode = "sales"
    show_tools = True
    print(f"\n  Current mode: {current_mode.upper()}")

    while True:
        print()
        try:
            user_input = input(f"  [{current_mode}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("  Goodbye!")
            break
        if user_input.lower() == "sales":
            current_mode = "sales"
            print("  Switched to SALES mode")
            continue
        if user_input.lower() == "property":
            current_mode = "property"
            print("  Switched to PROPERTY mode")
            continue
        if user_input.lower() == "reset":
            agent.reset(current_mode)
            print(f"  {current_mode} history cleared.")
            continue
        if user_input.lower() == "tools":
            show_tools = not show_tools
            print(f"  Tool display: {'ON' if show_tools else 'OFF'}")
            continue

        result = agent.chat(user_input, mode=current_mode)
        print(f"\n  RAEIPHI: {result['answer']}")

        if show_tools and result["tools_used"]:
            print(f"\n  Tools used:")
            for t in result["tools_used"]:
                print(f"    - {t['tool']}: {t['input']}")

        print(f"  ({result['total_time_ms']:.0f}ms)")


def run_demo(agent: RAEIPHIAgent):
    print("=" * 60)
    print("RAEIPHI — LangChain Agent Demo")
    print("=" * 60)

    demos = [
        ("sales", "How much would RAEIPHI cost for a two-bedroom Airbnb?"),
        ("sales", "Does it record any audio or video? My guests care about privacy."),
        ("sales", "I run 12 staffless hotel units across three cities. What would a deployment look like?"),
        ("property", "What's the current status of my downtown property?"),
        ("property", "Set my occupancy alert threshold to 8 people."),
    ]

    for i, (mode, question) in enumerate(demos):
        print(f"\n{'─' * 60}")
        print(f"  Demo {i + 1} [{mode.upper()} MODE]: \"{question}\"")
        print(f"{'─' * 60}")

        result = agent.chat(question, mode=mode)
        print(f"\n  Answer: {result['answer']}")

        if result["tools_used"]:
            print(f"\n  Tools used:")
            for t in result["tools_used"]:
                print(f"    - {t['tool']}: {t['input']}")

        print(f"  Timing: {result['total_time_ms']:.0f}ms")
        agent.reset()

    print(f"\n{'=' * 60}")
    print("DEMO COMPLETE")
    print(f"{'=' * 60}")
    print(f"\n  The agent used LangChain v1's tool-calling pattern:")
    print(f"  - Sales questions triggered search_documentation (RAG)")
    print(f"  - Property questions triggered check_property_status")
    print(f"  - Alert questions triggered manage_alerts")
    print(f"  - The agent decided WHICH tool to call based on the query")
    print(f"{'=' * 60}")


def main():
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    missing = []
    if not supabase_url: missing.append("SUPABASE_URL")
    if not supabase_key: missing.append("SUPABASE_SERVICE_KEY")
    if not groq_key: missing.append("GROQ_API_KEY")

    if missing:
        print(f"ERROR: Missing env vars: {', '.join(missing)}")
        sys.exit(1)

    agent = RAEIPHIAgent(supabase_url, supabase_key, groq_key)

    if "--demo" in sys.argv:
        run_demo(agent)
    else:
        interactive_chat(agent)


if __name__ == "__main__":
    main()
