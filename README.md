# AutoStream AI Sales Agent

A conversational AI agent built for **ServiceHive's Inflx internship assignment**.  
The agent acts as a smart sales assistant for **AutoStream** — a fictional SaaS platform for automated AI video editing.

---

## Features

- **Intent Detection** — Classifies each message as Greeting, Inquiry, or High-Intent
- **RAG Pipeline** — Answers pricing/feature questions from a local JSON knowledge base
- **State Management** — Retains full conversation history across 5–6+ turns via LangGraph
- **Tool Calling** — Triggers `capture_lead()` only after collecting all 3 required fields
- **Groq LLM** — Uses LLaMA 3.3 70B via Groq's free API (fast & capable)

---

## How to Run Locally

### 1. Clone / Download the project

```bash
cd autostream-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your free Groq API key

1. Go to [https://console.groq.com](https://console.groq.com)
2. Sign up / log in
3. Click **"API Keys"** → **"Create API Key"**
4. Copy the key (starts with `gsk_...`)

### 5. Set up your `.env` file

```bash
cp .env.example .env
```

Open `.env` in any text editor and paste your key:

```
GROQ_API_KEY=gsk_your_actual_key_here
```

### 6. Run the agent

```bash
python main.py
```

### Example conversation

```
You: Hi there!
Agent: Hello! I'm AutoStream's AI assistant. I can help you with pricing, features, or getting started. What would you like to know?

You: What are your pricing plans?
Agent: AutoStream offers two plans:
  • Basic Plan — $29/month: 10 videos/month, 720p resolution, basic editing tools, email support.
  • Pro Plan — $79/month: Unlimited videos, 4K resolution, AI captions, 24/7 priority support, custom branding.
  Annual billing saves you 2 months on either plan!

You: I want to try the Pro plan for my YouTube channel.
Agent: Amazing! Let's get you set up. What's your full name?

You: John Doe
Agent: Great, John! What's your email address?

You: john@example.com
Agent: Perfect! And which platform do you mainly create content for? (e.g. YouTube, Instagram, TikTok)

You: YouTube
Agent: You're all set, John! I've captured your details and our team will reach out to john@example.com shortly. Welcome to AutoStream! 🎬
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen because it provides structured **stateful graph execution** — essential for an agent that must track partial lead data (name, email, platform) across multiple conversation turns. Unlike simple LangChain chains, LangGraph lets you define explicit nodes (reasoning, tool execution) and conditional edges (should I call a tool or respond?), making the agent's control flow transparent and debuggable.

### How State is Managed

The agent uses a `TypedDict` called `AgentState` that persists throughout the entire conversation. It stores:
- `messages` — the full chat history (Human + AI + Tool messages), accumulated via LangGraph's `add_messages` reducer
- `lead_name`, `lead_email`, `lead_platform` — collected one at a time; all three must be present before `capture_lead` fires
- `intent` — the last classified user intent

Every turn, the full message history is re-injected into the LLM via a dynamic system prompt, giving the agent perfect memory across 5–6+ turns without any external database.

The graph has two nodes: `agent` (LLM call) and `tools` (tool execution). A conditional edge routes to tools only when the LLM outputs a tool call, then loops back to `agent` for the final natural-language response.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, you would use the **WhatsApp Business API (Meta Cloud API)**:

### Step-by-step integration

1. **Set up a Meta Developer App** at [developers.facebook.com](https://developers.facebook.com), enable WhatsApp, and get a phone number + access token.

2. **Register a Webhook URL** — Deploy this agent as a web server (e.g., FastAPI on Railway or Render). Meta will POST incoming WhatsApp messages to your `/webhook` endpoint.

3. **Verify the webhook** — Meta sends a `GET` request with a `hub.challenge` token. Your server must echo it back to confirm ownership.

4. **Process incoming messages**:
   ```python
   @app.post("/webhook")
   async def whatsapp_webhook(payload: dict):
       user_id = payload["entry"][0]["changes"][0]["value"]["messages"][0]["from"]
       text = payload["entry"][0]["changes"][0]["value"]["messages"][0]["text"]["body"]
       
       # Load state for this user (from Redis or in-memory dict keyed by phone number)
       state = get_or_create_state(user_id)
       state["messages"].append(HumanMessage(content=text))
       state = graph.invoke(state)
       save_state(user_id, state)
       
       reply = get_last_ai_message(state)
       send_whatsapp_message(user_id, reply)
   ```

5. **Send replies** using Meta's `/messages` API endpoint with your access token.

6. **Session state** — Store each user's `AgentState` in **Redis** (keyed by WhatsApp phone number) so conversations persist between webhook calls.

---

## Project Structure

```
autostream-agent/
├── agent/
│   ├── graph.py          # LangGraph agent — state, nodes, edges
│   └── rag.py            # RAG pipeline — knowledge base retrieval
├── tools/
│   └── lead_tools.py     # capture_lead tool + mock_lead_capture function
├── knowledge_base/
│   └── autostream_kb.json  # Pricing, features, policies, FAQs
├── main.py               # Entry point — interactive chat loop
├── requirements.txt      # Dependencies
├── .env.example          # Template for API key
└── README.md             # This file
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | LLaMA 3.3 70B via Groq (free) |
| Agent Framework | LangGraph |
| RAG | Keyword-based retrieval from local JSON |
| Tool Calling | LangChain `@tool` decorator |
| State | LangGraph `TypedDict` + `add_messages` |
| Runtime | Python 3.9+ |
