"""
rag.py — RAG (Retrieval-Augmented Generation) pipeline for AutoStream.

Strategy: Since the knowledge base is small and structured (JSON), we use
a lightweight in-memory approach with keyword/semantic matching instead of
a full vector DB. This keeps the project dependency-light while still
demonstrating RAG concepts cleanly.

For production scale: swap `retrieve()` internals with ChromaDB / FAISS + embeddings.
"""

import json
import re
from pathlib import Path
from typing import Optional


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def _load_kb() -> dict:
    with open(KB_PATH, "r") as f:
        return json.load(f)


def _build_chunks(kb: dict) -> list[dict]:
    """
    Converts the JSON knowledge base into flat text chunks for retrieval.
    Each chunk has a 'text' field and a 'source' label.
    """
    chunks = []

    # Company overview
    chunks.append({
        "source": "company_overview",
        "text": f"{kb['company']} — {kb['tagline']}",
        "keywords": ["autostream", "company", "what is", "about", "product"],
    })

    # Pricing plans
    for plan in kb["plans"]:
        features_text = ", ".join(plan["features"])
        chunks.append({
            "source": f"plan_{plan['name'].lower().replace(' ', '_')}",
            "text": (
                f"{plan['name']}: ${plan['price_monthly']}/month. "
                f"Features: {features_text}."
            ),
            "keywords": [
                "price", "pricing", "plan", "cost", "how much",
                plan["name"].lower(), "basic", "pro", "features",
                "resolution", "videos", "caption", "support",
            ],
        })

    # Policies
    policy_text = " ".join(kb["policies"])
    chunks.append({
        "source": "policies",
        "text": f"Company Policies: {policy_text}",
        "keywords": ["refund", "cancel", "policy", "support", "trial", "billing", "annual"],
    })

    # FAQs
    for faq in kb["faq"]:
        chunks.append({
            "source": "faq",
            "text": f"Q: {faq['question']} A: {faq['answer']}",
            "keywords": faq["question"].lower().split(),
        })

    return chunks


def retrieve(query: str, top_k: int = 3) -> str:
    """
    Retrieves the most relevant knowledge base chunks for a given query.

    Args:
        query: The user's question or message.
        top_k: Number of top chunks to return.

    Returns:
        A formatted string of relevant context passages.
    """
    kb = _load_kb()
    chunks = _build_chunks(kb)

    query_lower = query.lower()
    query_tokens = set(re.findall(r"\w+", query_lower))

    scored = []
    for chunk in chunks:
        keyword_set = set(chunk["keywords"])
        overlap = len(query_tokens & keyword_set)

        # Boost score for substring matches in the chunk text itself
        text_hits = sum(1 for token in query_tokens if token in chunk["text"].lower())

        score = overlap * 2 + text_hits
        if score > 0:
            scored.append((score, chunk))

    # Sort descending by score
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c["text"] for _, c in scored[:top_k]]

    if not top_chunks:
        return "No specific information found in the knowledge base for this query."

    return "\n\n".join(top_chunks)


def get_full_context() -> str:
    """Returns the entire knowledge base as a formatted string (for system prompt injection)."""
    kb = _load_kb()
    chunks = _build_chunks(kb)
    return "\n".join(c["text"] for c in chunks)
