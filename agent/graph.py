"""
graph.py — LangGraph-based Conversational AI Agent for AutoStream.

Architecture:
  - State: TypedDict with messages + lead collection fields
  - Nodes: agent_node (LLM reasoning) → tool_node (tool execution)
  - Edges: conditional routing based on whether the LLM wants to call a tool
  - Memory: full conversation history kept in state (persists across turns)
  - LLM: Groq (free tier) using llama-3.3-70b-versatile
"""

import os
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.rag import get_full_context
from tools.lead_tools import capture_lead


# ─── Agent State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    lead_name: Optional[str]
    lead_email: Optional[str]
    lead_platform: Optional[str]
    intent: Optional[str]


# ─── System Prompt ────────────────────────────────────────────────────────────

def build_system_prompt(state: AgentState) -> str:
    kb_context = get_full_context()

    lead_status = []
    if state.get("lead_name"):
        lead_status.append(f"Name: {state['lead_name']}")
    if state.get("lead_email"):
        lead_status.append(f"Email: {state['lead_email']}")
    if state.get("lead_platform"):
        lead_status.append(f"Platform: {state['lead_platform']}")

    lead_info = (
        "Lead info collected so far:\n" + "\n".join(lead_status)
        if lead_status
        else "No lead info collected yet."
    )

    return f"""You are AutoStream's intelligent sales assistant — friendly, concise, and helpful.
AutoStream provides automated AI-powered video editing tools for content creators.

KNOWLEDGE BASE (use this to answer questions accurately):
{kb_context}

INTENT CLASSIFICATION:
Classify every user message as one of:
  1. GREETING     — casual hello, how are you, etc.
  2. INQUIRY      — asking about pricing, features, policies, or general product questions
  3. HIGH_INTENT  — user expresses desire to sign up, try, purchase, or start a plan

LEAD COLLECTION RULES (follow strictly):
- Only begin collecting lead info when intent is HIGH_INTENT.
- Collect info ONE field at a time in this order: Name -> Email -> Platform.
- Do NOT ask for all three at once.
- {lead_info}
- Once you have all three (Name, Email, Platform), call the capture_lead tool immediately.
- NEVER call capture_lead before all three values are confirmed.

RESPONSE STYLE:
- Be warm, concise, and professional.
- For greetings: respond naturally, briefly mention you can help with AutoStream.
- For inquiries: answer using ONLY the knowledge base above.
- For high-intent: enthusiastically acknowledge, then start collecting info one field at a time.
- Keep responses under 100 words unless explaining pricing.
"""


# ─── LLM Setup ───────────────────────────────────────────────────────────────

def get_llm():
    """Returns a Groq LLaMA instance with the capture_lead tool bound."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=os.environ["GROQ_API_KEY"],
        max_tokens=1024,
        temperature=0.3,
    )
    return llm.bind_tools([capture_lead])


# ─── Graph Nodes ─────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> AgentState:
    llm = get_llm()
    system = SystemMessage(content=build_system_prompt(state))
    messages_with_system = [system] + state["messages"]
    response = llm.invoke(messages_with_system)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# ─── Build the Graph ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    tool_node = ToolNode([capture_lead])

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)

    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END},
    )
    builder.add_edge("tools", "agent")

    return builder.compile()
