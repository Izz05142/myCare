# MyCare — Maxis Agentic Customer Support Bot

A CLI-based agentic customer support bot built with LangChain and GPT-4o-mini (or Gemini). The agent autonomously decides which tool to call based on the customer's query — no hardcoded routing logic.

## Demo
> Add your demo video link here (YouTube unlisted)

## Architecture

```
User Input
    │
    ▼
LangChain Tool-Calling Agent (GPT-4o-mini)
    │
    ├── check_account_status(user_id)   → Returns account info from mock DB
    ├── get_faq_answer(topic)           → Searches FAQ knowledge base
    └── escalate_to_human(reason)       → Simulates ticket escalation
```

The agent uses LangChain's `create_tool_calling_agent` with a `ChatPromptTemplate` that maintains full conversation history across turns. The LLM decides at each step whether to call a tool or respond directly.

## Tech Stack
- Python 3.10+
- LangChain
- OpenAI GPT-4o-mini (or Google Gemini 1.5 Flash)
- CLI interface

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/mycare-bot
cd mycare-bot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API key
```

### 4. Run the bot
```bash
python agent.py
```

## Example Conversation

```
You: hi my data seems to be running low
MyCare: I can help with that! To check your exact data balance, could you provide your user ID?

You: its U001
MyCare: Here are your account details:
  Plan: Maxis Postpaid 100
  Status: Active
  Data Remaining: 12.5 GB
  ...

You: how do i activate roaming?
MyCare: FAQ — Roaming:
To activate international roaming, dial *100# or go to the Maxis app > Services > Roaming...

You: i want to speak to a human
MyCare: Your issue has been escalated to our support team.
  Ticket ID: TKT-48291
  A Maxis agent will contact you within 24 hours...
```

## Key Technical Decisions

**Why LangChain over CrewAI?**
LangChain's tool-calling agent is better suited for single-agent workflows with multiple tools. CrewAI shines for multi-agent pipelines where agents collaborate — overkill for this use case.

**Why tool calling over hardcoded routing?**
The agent dynamically decides which tool to invoke based on semantic understanding of the query. This is more robust than keyword matching and scales naturally to more tools.

**Why mock data instead of a real database?**
Keeps the demo self-contained and runnable without setup. In production, `check_account_status` would call a real CRM API.

## What I'd Improve With More Time
- Add a web UI (Streamlit or Next.js frontend)
- Connect to a real vector database (Pinecone/Chroma) for RAG-based FAQ retrieval instead of dictionary lookup
- Add more tools (e.g. `submit_complaint`, `check_outage_status`)
- Add memory persistence across sessions
- Implement proper logging and error handling for production use

## Author
Ahmad Izzat Ilmi bin Amirrudin
