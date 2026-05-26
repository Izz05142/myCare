from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
# Uncomment below and comment out OpenAI if using Gemini
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
 
load_dotenv()
 
# ─────────────────────────────────────────────
# Mock data
# ─────────────────────────────────────────────
MOCK_ACCOUNTS = {
    "U001": {"name": "Ahmad Izzat", "plan": "Maxis Postpaid 100", "status": "Active", "balance": "RM 45.20", "data_remaining": "12.5 GB"},
    "U002": {"name": "Siti Rahimah", "plan": "Maxis Prepaid", "status": "Suspended", "balance": "RM 0.00", "data_remaining": "0 GB"},
    "U003": {"name": "Raj Kumar", "plan": "Maxis Postpaid 50", "status": "Active", "balance": "RM 12.80", "data_remaining": "3.2 GB"},
}
 
MOCK_FAQS = {
    "roaming": "To activate international roaming, dial *100# or go to the Maxis app > Services > Roaming. Roaming passes start from RM30/day.",
    "bill": "Your bill is generated on the same date every month. You can view and pay your bill via the Maxis app or at www.maxis.com.my/billing.",
    "data": "To check your data balance, dial *100# or check the Maxis app. To buy a data add-on, go to Maxis app > Add-Ons.",
    "plan": "Available postpaid plans range from RM50 to RM250/month. Visit www.maxis.com.my/plans or call 1800-82-1123 for details.",
    "network": "If you are experiencing network issues, try restarting your device. If the issue persists, check coverage at coverage.maxis.com.my or contact support.",
    "sim": "To replace a lost or damaged SIM, visit any Maxis Store with your IC. A replacement fee of RM10 applies.",
}
 
# ─────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────
 
@tool
def check_account_status(user_id: str) -> str:
    """
    Check a customer's account status, plan, balance, and remaining data.
    Use this when the customer asks about their account, balance, data usage, or subscription.
    Args:
        user_id: The customer's user ID (e.g. U001)
    """
    account = MOCK_ACCOUNTS.get(user_id.strip().upper())
    if not account:
        return f"No account found for user ID '{user_id}'. Please verify the ID and try again."
    return (
        f"Account Details for {account['name']}:\n"
        f"  Plan: {account['plan']}\n"
        f"  Status: {account['status']}\n"
        f"  Outstanding Balance: {account['balance']}\n"
        f"  Data Remaining: {account['data_remaining']}"
    )
 
 
@tool
def get_faq_answer(topic: str) -> str:
    """
    Retrieve an answer from the FAQ knowledge base for common support topics.
    Use this when the customer asks a general question about services, plans, billing, roaming, network, or SIM.
    Args:
        topic: The topic to look up. One of: roaming, bill, data, plan, network, sim
    """
    topic = topic.strip().lower()
    for key, answer in MOCK_FAQS.items():
        if topic in key or key in topic:
            return f"FAQ — {key.capitalize()}:\n{answer}"
    return f"No FAQ found for '{topic}'. Available topics: {', '.join(MOCK_FAQS.keys())}"
 
 
@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalate the customer's issue to a human support agent.
    Use this when the issue is complex, the customer is frustrated, or the bot cannot resolve it.
    Args:
        reason: A brief summary of why escalation is needed
    """
    ticket_id = f"TKT-{abs(hash(reason)) % 100000:05d}"
    return (
        f"Your issue has been escalated to our support team.\n"
        f"Ticket ID: {ticket_id}\n"
        f"Reason logged: {reason}\n"
        f"A Maxis agent will contact you within 24 hours via your registered email or phone number."
    )
 
 
# ─────────────────────────────────────────────
# Agent setup
# ─────────────────────────────────────────────
 
SYSTEM_PROMPT = """You are MyCare, a helpful and professional customer support assistant for Maxis, a Malaysian telecommunications company.
 
Your job is to help customers with their queries by using the tools available to you.
 
Guidelines:
- Always be polite, clear, and concise
- Use check_account_status when the customer asks about their account, balance, or data — ask for their user ID first if not provided
- Use get_faq_answer for general questions about plans, billing, roaming, network, or SIM
- Use escalate_to_human if the issue is complex, the customer is upset, or you cannot resolve it with the available tools
- If a query is ambiguous, ask one clarifying question before proceeding
- Respond in the same language the customer uses (Malay or English)
"""
 
tools = [check_account_status, get_faq_answer, escalate_to_human]
 
 
def build_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    return llm.bind_tools(tools)
 
 
def process_tool_calls(tool_calls, messages):
    """Execute tool calls and append results to messages."""
    from langchain_core.messages import ToolMessage
    tool_map = {t.name: t for t in tools}
    for tc in tool_calls:
        tool_fn = tool_map.get(tc["name"])
        if tool_fn:
            result = tool_fn.invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return messages
 
 
def chat(agent, messages):
    """Run one turn, handling tool calls in a loop."""
    while True:
        response = agent.invoke(messages)
        messages.append(response)
        
        # If the model didn't call a tool, it has provided its final response
        if not response.tool_calls:
            return response.content, messages
        
        # Process tool calls and append ToolMessages to the history
        messages = process_tool_calls(response.tool_calls, messages)
        
        # IMPORTANT: The loop continues here! 
        # On the next iteration, agent.invoke(messages) is called again, 
        # sending the tool results back to Gemini so it can answer the user.
 
 
# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
 
def main():
    print("=" * 50)
    print("  MyCare — Maxis Agentic Support Bot")
    print("  Type 'exit' to quit")
    print("=" * 50)
 
    agent = build_agent()
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
 
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("MyCare: Thank you for contacting Maxis. Have a great day!")
            break
 
        messages.append(HumanMessage(content=user_input))
 
        try:
            reply, messages = chat(agent, messages)
            
            # Clean up the response if it comes back as a list of dictionaries
            if isinstance(reply, list):
                clean_reply = "".join([item.get("text", "") for item in reply if isinstance(item, dict)])
            else:
                clean_reply = str(reply)
                
            print(f"\nMyCare: {clean_reply.strip()}")
        except Exception as e:
            print(f"\nMyCare: Sorry, something went wrong. ({e})")
 
 
if __name__ == "__main__":
    main()
