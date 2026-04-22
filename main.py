"""
main.py — Entry point for the AutoStream Conversational AI Agent.

Run:
    python main.py

The agent runs in your terminal as an interactive chat loop.
Type 'exit' or 'quit' to end the session.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import build_graph, AgentState

load_dotenv()


BANNER = """
╔══════════════════════════════════════════════════════╗
║          AutoStream AI Sales Assistant               ║
║    Powered by Groq (LLaMA 3.3 70B) + LangGraph      ║
╚══════════════════════════════════════════════════════╝
  Type your message below. Type 'exit' to quit.
"""


def extract_text(content) -> str:
    """Safely extract text from an AIMessage content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content)


def run():
    # Validate API key
    if not os.environ.get("GROQ_API_KEY"):
        print("❌  GROQ_API_KEY not set. Please add it to your .env file.")
        print("    Get a free key at: https://console.groq.com")
        return

    graph = build_graph()
    print(BANNER)

    # Initial state
    state: AgentState = {
        "messages": [],
        "lead_name": None,
        "lead_email": None,
        "lead_platform": None,
        "intent": None,
    }

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋  Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Agent: Thanks for chatting! Have a great day creating content. 🎬")
            break

        # Add user message to state
        state["messages"].append(HumanMessage(content=user_input))

        # Run the graph
        state = graph.invoke(state)

        # Print the latest assistant message
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage):
                text = extract_text(msg.content)
                if text:
                    print(f"\nAgent: {text}\n")
                    break


if __name__ == "__main__":
    run()
