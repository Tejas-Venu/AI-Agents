# Topic 7 - A2A Trivia Tournament

---

## Table of Contents

1. [Project Directory](#project-directory)   
2. [Agent Template](#agent-template)  
3. [System Test Script](#system-test-script)  
4. [Trivia Tournament Script](#trivia-tournament-script)  
5. [Discussion Questions](#discussion-questions)  
6. [Conclusion](#conclusion)  

---

# Project Directory

```text
Topic7MCP/
├── A2A/
│   ├── a2a_agent_template.py
│   ├── a2a_test.py
│   ├── a2a_trivia.py
│   └── test_ngrok.sh
└── MCP/
```
---
# Agent Template

`a2a_agent_template.py` is the main file used to build a student agent.

## What the template does

- Starts a FastAPI app
- Exposes an Agent Card at `/.well-known/agent.json`
- Exposes a task endpoint at `/task`
- Detects the public ngrok URL automatically
- Registers the agent with the class registry
- Uses GPT-4o mini to answer questions in `handle_task()`

---

# System Test Script
`a2a_test.py` is a local test harness for the A2A pipeline.

## What it tests
- Starts a local registry
- Starts fake agents on separate ports
- Registers agents with the registry
- Lists all registered agents
- Sends a task to one agent
- Broadcasts trivia questions to all agents
- Checks health endpoints

## Purpose
This script verifies that the registry, agent cards, task endpoints, broadcast flow and health checks all work end-to-end before the real tournament.

---

# Trivia Tournament Script

`a2a_trivia.py` runs the full tournament.

## What it does

- Loads 24 trivia questions across 6 categories
- Broadcasts each question to all registered agents
- Optionally routes questions to the best-matching agents using TF-IDF similarity
- Uses GPT-4o mini to judge whether each answer is correct
- Optionally picks the funniest wrong answer
- Prints a final leaderboard


## Smart routing

The script can route questions to the top matching agents using TF-IDF cosine similarity over:
- agent name
- agent description
- skill name
- skill description
This allows the tournament to prioritize the most relevant agents instead of broadcasting to everyone.

---

# Discussion Questions

## MCP vs A2A: How is sending a task to another agent different from calling an MCP tool? What can an agent do that a tool cannot?

Sending a task to another agent is different from calling an MCP tool because an A2A agent is a whole autonomous participant with its own prompt, reasoning, and response style. An MCP tool is just a callable capability that returns data, while an agent can interpret the task, decide how to answer, and maintain its own identity across requests.

## Discovery: We used a central registry. What are the alternatives? What are the tradeoffs of centralized vs decentralized discovery?

A central registry makes discovery simple because everyone checks one place for active agents, but it also creates a single point of failure. Decentralized discovery could rely on peer exchange, gossip, or direct configuration, which may scale better in some settings but is harder to coordinate and more difficult to manage consistently.

## System prompts as strategy: How much did the system prompt matter for scoring? Could you craft a prompt that is good at all categories while still being funny on off-topic questions?

The system prompt matters a lot because it determines whether the agent stays on-topic and whether it produces funny off-topic responses instead of factual ones. A prompt can work across all categories if it is very explicit about in-scope behavior, out-of-scope behavior, and tone, but it still needs careful testing because models can sometimes drift.

## Smart routing: TF-IDF matched questions to agents based on text overlap. What would happen with semantic embeddings instead? What if agents could self-report confidence?

TF-IDF matching works by looking for word overlap, so it is strong when the question and the agent description use similar vocabulary. Semantic embeddings would likely do better because they can capture meaning and synonyms even when the exact words do not match, and self-reported confidence could help the router avoid low-confidence assignments.

## Trust and reliability: In a real multi-agent system, how would you handle an agent that returns bad data? What if an agent is slow or goes offline mid-task?

A real multi-agent system needs validation because not every agent will answer correctly or even respond on time. If an agent is slow or offline, the system should fall back to another agent or continue without it, and if an agent returns bad data, the judge or orchestrator should verify the response before scoring or trusting it.

## Scaling: What would break if there were 1,000 agents instead of 20? What architectural changes would you need?

With 1,000 agents, broadcasting every question to everyone would become too expensive and too slow. The architecture would need better indexing, better routing, asynchronous fan-out, health monitoring, and probably semantic retrieval instead of simple TF-IDF matching to keep the system manageable.

---

# Conclusion

This project demonstrates:
The final system combines:
- an editable agent template
- a local testing harness
- a tournament runner
- structured scoring and leaderboard logic
into a full classroom A2A trivia competition.