# Agent Tool Use — Portfolio README
---

## Table of Contents

1. [Overview](#overview)  
2. [Task 1 — Sequential vs Parallel Execution (With and Without Ollama)](#task-1--sequential-vs-parallel-execution-with-and-without-ollama)  
   - [Without Ollama — Sequential Execution](#without-ollama--sequential-execution)  
   - [Without Ollama — Parallel Execution](#without-ollama--parallel-execution)  
   - [With Ollama — Sequential Execution](#with-ollama--sequential-execution)  
   - [With Ollama — Parallel Execution](#with-ollama--parallel-execution)  
3. [Task 2 — OpenAI Setup and API Key Handling](#task-2--openai-setup-and-api-key-handling)  
4. [Task 3 — Manual Tool Handling & Custom Calculator](#task-3--manual-tool-handling--custom-calculator)  
   - [Supported Features](#supported-features)  
   - [Strategy for Enforcing Tool Use](#strategy-for-enforcing-tool-use)  
5. [Task 4 — LangGraph Tool Handling](#task-4--langgraph-tool-handling)  
   - [Example: Letter Counting](#example-letter-counting)  
   - [Example: Multi-Step Tool Chaining](#example-multi-step-tool-chaining)  
6. [Task 5 — Persistent LangGraph Conversation](#task-5--persistent-langgraph-conversation)  
   - [Features Implemented](#features-implemented)  
7. [Task 6 — Parallelization Opportunity](#task-6--where-is-there-an-opportunity-for-parallelization-in-your-agent-that-is-not-yet-being-taken-advantage-of)  
8. [Conclusion](#conclusion)  


---

## Overview

This project implements and evaluates:

- Llama 3.2-1B model evaluations on different MMLU topics  
- Sequential versus parallel execution timing  
- Ollama-based local model serving  
- OpenAI GPT-4o Mini API usage  
- Manual tool handling with a custom calculator  
- LangGraph-based multi-tool agent orchestration  
- Persistent conversation with checkpointing and recovery  
- Identification of parallelization opportunities  

---

# Task 1 — Sequential vs Parallel Execution (With and Without Ollama)

Two modified versions of the evaluation program were created:

- One evaluating **ASTRONOMY**
- One evaluating **BUSINESS ETHICS**

Each was executed sequentially and then in parallel. The experiments were repeated after modifying the programs to use Ollama.

---

## Without Ollama — Sequential Execution

### ASTRONOMY Evaluation
- Result: **72/152 = 47.37%**
- Completed in **13.4 seconds**

### BUSINESS ETHICS Evaluation
- Result: **39/100 = 39.00%**

**Total real time:** `1m 9.236s`

---

## Without Ollama — Parallel Execution

### Results
- BUSINESS ETHICS: **39/100 = 39.00%**
- ASTRONOMY: **72/152 = 47.37%**

**Total real time:** `0m 48.855s`

### Observation

Parallel execution reduced total wall-clock time compared to sequential execution. CPU usage increased, but overall completion time decreased.

---

## With Ollama — Sequential Execution

### ASTRONOMY Evaluation (Ollama)
- Result: **27/152 = 17.76%**
- Completed in **52.7 seconds**

### BUSINESS ETHICS Evaluation (Ollama)
- Result: **30/100 = 30.00%**

**Total real time:** `1m 28.845s`

---

## With Ollama — Parallel Execution

### Results
- BUSINESS ETHICS: **30/100 = 30.00%**
- ASTRONOMY: **27/152 = 17.76%**

**Total real time:** `1m 22.489s`

### Observation

Using Ollama increased execution time in this environment compared to the non-Ollama runs. However, parallel execution still reduced overall wall time compared to sequential Ollama execution.

Performance depends on:

- Hardware configuration (CPU vs GPU)
- Model loading overhead
- Caching behavior

---

# Task 2 — OpenAI Setup and API Key Handling

The OpenAI API key was stored securely using environment variables (or Colab’s secret manager). It was never committed to version control.

- The OpenAI client initialization creates an authenticated connection using the secret key stored in the environment.
- The chat completion call sends a message to the GPT-4o Mini model and returns the model’s response.
- The `max_tokens` parameter limits response length.

This confirms that:

- The environment is properly configured  
- The API connection works correctly  
- The key is securely managed  

---

# Task 3 — Manual Tool Handling & Custom Calculator

A custom calculator tool was implemented using structured JSON input and output.

- Input is parsed safely
- Expressions are evaluated using a restricted mathematical evaluator
- Output is returned in structured format

## Supported Features

- Basic arithmetic
- Trigonometric functions
- Geometric functions:
  - Area of circle
  - Area of rectangle
  - Area of triangle
  - Additional standard geometric calculations

## Strategy for Enforcing Tool Use

To force the LLM to use the calculator tool instead of computing internally:

- Explicit instructions were placed in the system prompt
- The model was required to use the calculator for **all mathematical operations**, including simple arithmetic

This significantly reduced the model’s tendency to compute answers directly and improved tool invocation consistency.

---

# Task 4 — LangGraph Tool Handling

Three tools were integrated into the LangGraph agent:

1. Calculator  
2. Letter counting tool  
3. Custom text statistics tool  

## Example: Letter Counting

Query:
> “How many s are in Mississippi riverboats?”

The system correctly invokes the letter counting tool to compute occurrences.

## Example: Multi-Step Tool Chaining

Query:
> “What is the sin of the difference between the number of i’s and s’s in Mississippi riverboats?”

Execution steps:

- Two calls to the letter counting tool  
- One call to the calculator tool  
- Sequential chaining handled by the LangGraph orchestration loop  

The agent successfully performed multi-tool reasoning within the conversation.

---

# Task 5 — Persistent LangGraph Conversation

The agent was rewritten using LangGraph nodes instead of a simple Python loop.

## Features Implemented

- Persistent conversation state  
- SQLite checkpointing  
- Recovery after interruption  
- Idempotent tool execution  
- Reinsertion of missing tool results on resume  
- History truncation to prevent token explosion  

The system successfully maintained conversation context across runs and recovered safely from interrupted states.

---

# Task 6 — where is there an opportunity for parallelization in your agent that is not yet being taken advantage of?

A clear parallelization opportunity exists inside the `call_tools` node. When the LLM returns multiple `tool_calls` in a single assistant message, the current implementation executes them sequentially in a `for` loop, even though many of these calls are independent (e.g., counting letters twice before a calculator call). These tool invocations could be executed concurrently using a thread pool or async execution, then appended to history once all results are complete. The same sequential pattern also appears in the resume-repair logic where pending tool calls are executed one-by-one, which could similarly be parallelized to reduce latency.

---

# Conclusion

This project demonstrates:

- Performance differences between sequential and parallel execution  
- Performance differences between direct model usage and Ollama  
- Effective enforcement of tool usage in LLMs  
- Multi-tool orchestration with LangGraph  
- Persistent agent design with checkpointing and recovery  
- Identification of real-world optimization opportunities  

The final system integrates:

- Evaluation experiments  
- Tool usage enforcement  
- Agent orchestration  
- Performance analysis  

All within a structured and recoverable framework.