"""
tools.py — Tool definitions for the AutoStream AI Agent.
Includes the mock lead capture function and its LangChain tool wrapper.
"""

import json
from datetime import datetime
from langchain_core.tools import tool


# ─── Mock Lead Capture ────────────────────────────────────────────────────────

def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Simulates saving a qualified lead to a CRM backend.
    In production, this would POST to a real CRM API (HubSpot, Salesforce, etc.)
    """
    lead = {
        "name": name,
        "email": email,
        "platform": platform,
        "captured_at": datetime.utcnow().isoformat() + "Z",
        "status": "qualified",
        "source": "AutoStream AI Agent",
    }

    print("\n" + "=" * 55)
    print("  ✅  LEAD CAPTURED SUCCESSFULLY")
    print("=" * 55)
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Time     : {lead['captured_at']}")
    print("=" * 55 + "\n")

    return lead


# ─── LangChain Tool Wrapper ───────────────────────────────────────────────────

@tool
def capture_lead(name: str, email: str, platform: str) -> str:
    """
    Call this tool ONLY when the user has explicitly expressed intent to sign up
    or try AutoStream AND has provided their name, email, and creator platform.
    Do NOT call this prematurely. Collect all three values first through conversation.

    Args:
        name: Full name of the potential customer.
        email: Email address of the potential customer.
        platform: The creator platform they use (e.g., YouTube, Instagram, TikTok).

    Returns:
        A confirmation string with the captured lead details.
    """
    result = mock_lead_capture(name, email, platform)
    return (
        f"Lead captured successfully! Here's a summary:\n"
        f"- Name: {result['name']}\n"
        f"- Email: {result['email']}\n"
        f"- Platform: {result['platform']}\n"
        f"- Captured at: {result['captured_at']}"
    )
