"""
A2A System Test Script
=======================
This script tests the full A2A pipeline without needing ngrok or real students.
It spins up a registry and two fake agents locally, registers them, and runs
a trivia round to verify everything works end-to-end.

USAGE:
  python a2a_test.py

DEPENDENCIES:
  pip install fastapi uvicorn requests openai python-dotenv

WHAT IT DOES:
  1. Starts the registry server on port 8001
  2. Starts two fake agents on ports 8002 and 8003
  3. Registers both agents with the registry
  4. Lists all registered agents
  5. Sends a question to one specific agent
  6. Broadcasts a trivia question to all agents
  7. Prints all results

You can also use this script to test against real student agents â€”
just skip the fake agent setup and point at the real registry.
"""

import os
import time
import json
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# âš™ï¸  CONFIGURATION
# =============================================================================

REGISTRY_URL = "http://localhost:8001"

# Set to True to start local fake agents for testing.
# Set to False if you want to test against real running agents.
USE_FAKE_AGENTS = True

# If USE_FAKE_AGENTS is True, these will be started automatically.
# If False, the script assumes the registry and agents are already running.
FAKE_AGENTS = [
    {
        "name": "History Agent",
        "description": "Answers questions about world history",
        "skills": [{"id": "history", "name": "History Q&A", "description": "World history expert"}],
        "port": 8002,
    },
    {
        "name": "Science Agent",
        "description": "Answers questions about science and nature",
        "skills": [{"id": "science", "name": "Science Q&A", "description": "Science and nature expert"}],
        "port": 8003,
    },
]

TRIVIA_QUESTIONS = [
    "What year did the Berlin Wall fall?",
    "What is the chemical symbol for gold?",
    "Who painted the Mona Lisa?",
]

# =============================================================================
# ðŸ¤–  FAKE AGENT SERVER (for local testing without ngrok)
# =============================================================================

def start_fake_agent(agent_config: dict):
    """Start a minimal FastAPI agent in a background thread."""
    from fastapi import FastAPI, Request
    import uvicorn

    fake_app = FastAPI()
    name = agent_config["name"]

    @fake_app.get("/.well-known/agent.json")
    async def agent_card():
        return {
            "name": name,
            "description": agent_config["description"],
            "url": f"http://localhost:{agent_config['port']}",
            "skills": agent_config["skills"],
        }

    @fake_app.post("/task")
    async def handle_task(request: Request):
        body = await request.json()
        question = body.get("question", "")
        # Simple canned response â€” in real usage this would call an LLM
        return {
            "agent": name,
            "answer": f"[{name}] I received your question: '{question}'. "
                      f"Here is a placeholder answer from {name}.",
        }

    @fake_app.get("/health")
    async def health():
        return {"status": "ok", "agent": name}

    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={"app": fake_app, "host": "0.0.0.0", "port": agent_config["port"], "log_level": "error"},
        daemon=True,
    )
    thread.start()
    return thread


# =============================================================================
# ðŸ§ª  TEST FUNCTIONS
# =============================================================================

def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_register_agents():
    """Register all fake agents with the registry."""
    separator("TEST 1: Register Agents")

    for agent in FAKE_AGENTS:
        url = f"http://localhost:{agent['port']}"
        resp = requests.post(f"{REGISTRY_URL}/register", json={
            "name": agent["name"],
            "url": url,
            "description": agent["description"],
            "skills": agent["skills"],
        })
        data = resp.json()
        print(f"  âœ… Registered '{agent['name']}' â†’ {url}")
        print(f"     Registry now has {data['total_agents']} agent(s)")


def test_list_agents():
    """List all registered agents."""
    separator("TEST 2: List All Agents")

    resp = requests.get(f"{REGISTRY_URL}/agents")
    data = resp.json()
    print(f"  Found {data['count']} agent(s):\n")

    for agent in data["agents"]:
        skills = ", ".join(s["name"] for s in agent["skills"])
        print(f"  ðŸ¤– {agent['name']}")
        print(f"     Description: {agent['description']}")
        print(f"     Skills: {skills}")
        print(f"     URL: {agent['url']}")
        print(f"     Status: {agent['status']}")
        print()


def test_filter_by_skill():
    """Test skill-based filtering."""
    separator("TEST 3: Filter Agents by Skill")

    for keyword in ["history", "science", "cooking"]:
        resp = requests.get(f"{REGISTRY_URL}/agents", params={"skill": keyword})
        data = resp.json()
        names = [a["name"] for a in data["agents"]]
        print(f"  Skill '{keyword}': {names if names else 'no matches'}")


def test_fetch_agent_cards():
    """Fetch the Agent Card directly from each agent."""
    separator("TEST 4: Fetch Agent Cards")

    resp = requests.get(f"{REGISTRY_URL}/agents")
    for agent in resp.json()["agents"]:
        card_url = f"{agent['url']}/.well-known/agent.json"
        card_resp = requests.get(card_url)
        card = card_resp.json()
        print(f"  ðŸ“‹ Card from {card['name']}:")
        print(f"     {json.dumps(card, indent=6)}")
        print()


def test_send_to_one():
    """Send a question to a specific agent."""
    separator("TEST 5: Send Task to One Agent")

    target = FAKE_AGENTS[0]["name"]
    question = "When was the Declaration of Independence signed?"

    print(f"  Sending to '{target}': {question}\n")

    resp = requests.post(f"{REGISTRY_URL}/send", json={
        "name": target,
        "question": question,
        "sender": "test-script",
    })
    data = resp.json()
    print(f"  ðŸ“¤ Response from {data['agent']}:")
    print(f"     {data['answer']}")


def test_broadcast_trivia():
    """Broadcast trivia questions to all agents."""
    separator("TEST 6: Broadcast Trivia Round")

    for i, question in enumerate(TRIVIA_QUESTIONS, 1):
        print(f"  ðŸ“¢ Question {i}: {question}\n")

        resp = requests.post(f"{REGISTRY_URL}/broadcast", json={
            "question": question,
            "sender": "trivia-master",
        })
        data = resp.json()

        for r in data["responses"]:
            status = "âœ…" if r["status"] == "success" else "âŒ"
            print(f"     {status} {r['agent']}: {r.get('answer', r.get('error', 'no response'))}")

        print()


def test_health_check():
    """Check the health of the registry and all agents."""
    separator("TEST 7: Health Checks")

    # Registry health
    resp = requests.get(f"{REGISTRY_URL}/health")
    data = resp.json()
    print(f"  Registry: {data['status']} ({data['online_agents']}/{data['total_agents']} online)")

    # Individual agent health
    agents_resp = requests.get(f"{REGISTRY_URL}/agents")
    for agent in agents_resp.json()["agents"]:
        try:
            h = requests.get(f"{agent['url']}/health", timeout=3)
            print(f"  {agent['name']}: {h.json()['status']}")
        except Exception as e:
            print(f"  {agent['name']}: âŒ unreachable ({e})")


# =============================================================================
# ðŸ  MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  ðŸ§ª  A2A System Test")
    print("=" * 60)

    if USE_FAKE_AGENTS:
        # Start the registry
        print("\n  Starting registry on port 8001...")
        import subprocess, sys
        registry_proc = subprocess.Popen(
            [sys.executable, "a2a_registry.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)

        # Start fake agents
        for agent in FAKE_AGENTS:
            print(f"  Starting fake agent '{agent['name']}' on port {agent['port']}...")
            start_fake_agent(agent)
        time.sleep(1)

        print("\n  All servers running. Starting tests...\n")
    else:
        print("\n  Using existing registry and agents. Starting tests...\n")

    try:
        test_register_agents()
        test_list_agents()
        test_filter_by_skill()
        test_fetch_agent_cards()
        test_send_to_one()
        test_broadcast_trivia()
        test_health_check()

        separator("ALL TESTS COMPLETE âœ…")
        print("  The registry and agent communication pipeline is working.")
        print("  Open http://localhost:8001 in your browser to see the dashboard.")
        print()

    finally:
        if USE_FAKE_AGENTS:
            registry_proc.terminate()
            print("  Cleaned up registry process.")