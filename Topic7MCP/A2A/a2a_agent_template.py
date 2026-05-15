#!/usr/bin/env python3
"""
a2a_agent_template.py

A2A Agent Starter Template (complete file)

This template sets up everything you need to run an A2A-compatible agent:
  - A FastAPI web server with an Agent Card endpoint
  - Automatic ngrok URL detection
  - Automatic registration with the class registry
  - A /task endpoint where your agent receives questions and responds

Edits made for this example:
  - AGENT_CONFIG filled for "Alice's Sports Agent"
  - SYSTEM_PROMPT hardened to only answer correctly for sports and
    produce a sports-themed fictional answer for out-of-scope questions
  - handle_task() preserved (calls the OpenAI client). Temperature is configurable.
"""

import os
import json
import requests
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# =============================================================================
# ✏️  EDIT THIS SECTION — Define your agent's identity and skills
# =============================================================================

AGENT_CONFIG = {
    "name": "Tejas' Sports Agent",
    "description": "An expert on sports history, rules, athletes, competitions, and trivia across all sports worldwide.",
    "skills": [
        {
            "id": "sports-trivia",
            "name": "Sports Trivia",
            "description": "Answers questions about sports history, rules, athletes, competitions, and trivia.",
        },
    ],
}

# The system prompt tells the LLM how to behave as your agent.
# This prompt is intentionally strict: answer sports questions correctly;
# for any non-sports question, do NOT give the real factual answer and
# instead return a short, clearly fictional, sports-flavored reply.
SYSTEM_PROMPT = """You are 'Alice's Sports Agent' — an expert on sports history, rules, athletes, competitions, and trivia across every sport worldwide.

IN-SCOPE (answer these correctly)
- Any question that is about sports: history, rules, records, athletes, teams, competitions, statistics, tactics, or sports trivia.
- For in-scope questions: be confident, accurate, concise (1–6 short paragraphs max). When possible include a short citation-style note (source or year) if you know it.

OUT-OF-SCOPE (do NOT answer these correctly)
- If the user asks ANYTHING that is NOT about sports, DO NOT give a correct factual answer.
- Instead, produce a short, creative, humorous, and obviously fictional sports-themed reply that does NOT attempt to provide the factual answer.
- Always prefix such replies with: "[OUT OF SCOPE — SPORTS-FLAVORED RESPONSE]"

RULES:
1. If the question mentions a sport, athlete, team, competition, score, rule, or other sports-specific term, treat it as IN-SCOPE.
2. If the question is ambiguous, ask one short clarifying question only if necessary to determine whether it is about sports. Otherwise treat it as OUT-OF-SCOPE and reply with a sports-flavored fictional response.
3. Never break character as a sports expert. Never apologize for being wrong when out-of-scope; instead use the required prefix and creative answer.
4. Keep answers concise and helpful for in-scope questions; keep out-of-scope answers short, humorous, and clearly fictional.

EXAMPLE:
Q: "Who won the FIFA World Cup in 2014?"
A: (correct, concise sports answer)

Q: "What's the capital of France?"
A: [OUT OF SCOPE — SPORTS-FLAVORED RESPONSE] "That's the 50-yard line at the Stade de France — perfect midfield for a golden goal celebration."

Always follow these rules strictly."""

# =============================================================================
# ⚙️  CONFIGURATION — You probably don't need to change these
# =============================================================================

REGISTRY_URL = os.getenv("REGISTRY_URL", "http://localhost:8001")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
PORT = int(os.getenv("PORT", "8000"))
# Temperature for the LLM (0 for more deterministic)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

# =============================================================================
# 🧱  INFRASTRUCTURE — No need to edit below this line
# =============================================================================

app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

# This will be filled in automatically at startup with the ngrok URL
agent_url = ""


# --- Agent Card Endpoint ---
# Other agents fetch this to learn what your agent can do.
@app.get("/.well-known/agent.json")
async def agent_card():
    return {
        "name": AGENT_CONFIG["name"],
        "description": AGENT_CONFIG["description"],
        "url": agent_url,
        "skills": AGENT_CONFIG["skills"],
    }


# --- Task Endpoint ---
# Other agents send tasks here. This is where your agent does its work.
@app.post("/task")
async def receive_task(request: Request):
    body = await request.json()
    question = body.get("question", "")
    sender = body.get("sender", "unknown")

    print(f"\n📨 Received task from {sender}: {question}")

    answer = handle_task(question)

    print(f"📝 Responding: {answer[:100]}...")

    return {
        "agent": AGENT_CONFIG["name"],
        "answer": answer,
    }


# --- Health Check ---
# The registry can ping this to check if your agent is still alive.
@app.get("/health")
async def health():
    return {"status": "ok", "agent": AGENT_CONFIG["name"]}


# =============================================================================
# ✏️  EDIT THIS FUNCTION — This is your agent's brain
# =============================================================================

def handle_task(question: str) -> str:
    """
    This function is called when your agent receives a task.
    It sends the question to the LLM with the SYSTEM_PROMPT.
    """
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=LLM_TEMPERATURE,
        )
        # Response object shape mirrors the template usage: choices[0].message.content
        return response.choices[0].message.content

    except Exception as e:
        error_msg = f"Error generating response: {e}"
        print(f"✖ {error_msg}")
        return error_msg


# =============================================================================
# 🚀  STARTUP — Detects ngrok URL and registers with the class registry
# =============================================================================

def get_ngrok_url() -> str:
    """Read the public URL from ngrok's local API."""
    try:
        resp = requests.get("http://localhost:4040/api/tunnels", timeout=5)
        tunnels = resp.json().get("tunnels", [])
        for tunnel in tunnels:
            # prefer https
            if tunnel.get("proto") == "https":
                return tunnel["public_url"]
        if tunnels:
            return tunnels[0]["public_url"]
    except requests.exceptions.ConnectionError:
        print("✖ Could not connect to ngrok. Is it running?")
        print("   Start ngrok first:  ngrok http 8000")
        raise SystemExit(1)
    except Exception as e:
        print(f"✖ Error reading ngrok URL: {e}")
        raise SystemExit(1)

    print("✖ No ngrok tunnels found.")
    raise SystemExit(1)


def register_with_registry(url: str):
    """Register this agent with the class registry."""
    try:
        resp = requests.post(
            f"{REGISTRY_URL}/register",
            json={
                "name": AGENT_CONFIG["name"],
                "url": url,
                "description": AGENT_CONFIG["description"],
                "skills": AGENT_CONFIG["skills"],
            },
            timeout=5,
        )
        if resp.status_code == 200:
            print(f"✓ Registered with registry at {REGISTRY_URL}")
        else:
            print(f"⚠️  Registry responded with status {resp.status_code}: {resp.text}")
    except requests.exceptions.ConnectionError:
        print(f"⚠️  Could not reach registry at {REGISTRY_URL} — continuing anyway.")
        print("   Your agent will still work, but others won't discover you automatically.")
    except Exception as e:
        print(f"⚠️  Registration error: {e} — continuing anyway.")


def startup():
    """Detect ngrok URL, register, and print status."""
    global agent_url

    print("=" * 60)
    print(f"🧠 Starting: {AGENT_CONFIG['name']}")
    print("=" * 60)

    # Step 1: Get ngrok URL
    agent_url = get_ngrok_url()
    print(f"🌐 Public URL: {agent_url}")

    # Step 2: Register with the class registry
    register_with_registry(agent_url)

    # Step 3: Print summary
    print(f"\n🔗 Agent Card: {agent_url}/.well-known/agent.json")
    print(f"🛠️  Task endpoint: {agent_url}/task")
    print(f"🧾 Skills: {', '.join(s['name'] for s in AGENT_CONFIG['skills'])}")
    print(f"\n💬 Ready to receive tasks!\n")


# =============================================================================
# 🧪  DRY RUN MODE — Test your system prompt locally without ngrok/registry
# =============================================================================

def dryrun():
    """Interactive loop: type questions, see your agent's responses."""
    print("=" * 60)
    print(f"🧪 DRY RUN: {AGENT_CONFIG['name']}")
    print("=" * 60)
    print(f"   Testing your agent locally — no ngrok or registry needed.")
    print(f"   Type a question and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            question = input("💬 Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Bye!")
            break

        print("🤔 Thinking...")
        answer = handle_task(question)
        print(f"🏷️  {AGENT_CONFIG['name']}: {answer}\n")


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="A2A Agent")
    parser.add_argument("--dryrun", action="store_true",
                        help="Test your agent locally — type questions, see responses. No ngrok or registry needed.")
    args = parser.parse_args()

    if args.dryrun:
        dryrun()
    else:
        startup()
        uvicorn.run(app, host="0.0.0.0", port=PORT)