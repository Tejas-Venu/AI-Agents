# LangGraph Multi-Agent Development Project

This project incrementally builds a **LangGraph-based multi-agent conversational system** using Hugging Face LLMs (Llama and Qwen).

It demonstrates:
- Node design and routing  
- Conditional execution  
- Parallel execution  
- Model switching  
- Message API integration  
- Multi-agent shared history  
- Checkpointing and crash recovery  

---

# Task 1 : Running and Understanding `langgraph_simple_llama_agent.py`

The initial program:
- Loads `meta-llama/Llama-3.2-1B-Instruct`
- Wraps it using `HuggingFacePipeline`
- Defines a `StateGraph`
- Implements:
  - `get_user_input`
  - `call_llm`
  - `print_response`
  - A router function
- Uses conditional edges and a loop-back structure

## Architecture

```
START
  ↓
get_user_input
  ↓ (conditional)
call_llm
  ↓
print_response
  ↓
get_user_input (loop)
```

## Verbose / Quiet Mode

Modified the program so that:
- Typing `verbose` enables tracing
- Typing `quiet` disables tracing

Each node checks:

```python
if state.get("verbose", False):
    print("[TRACE] Entering node")
```

This allows runtime debugging without restarting the agent.

---

# Task 2 : Handling Empty Input

## Observed Behavior

On the first empty input, the LLM was still invoked despite receiving a blank prompt and it generated a random, unrelated output. On the second consecutive empty input, the model began hallucinating a continuation of the conversation, fabricating context as if an ongoing dialogue existed. This behavior highlights how smaller models can attempt to infer missing context and continue generating responses even when provided with insufficient or malformed input.

This reveals that smaller models such as `Llama-3.2-1B-Instruct`:
- Attempt to continue conversation when given weak prompts.
- Are less robust to malformed inputs.


## Proper LangGraph Fix

Implemented a 3-way conditional branch:

```
get_user_input
  ├── END
  ├── get_user_input (self-loop if empty)
  └── call_llm
```

Router logic:

```python
if should_exit:
    return END
if not user_input.strip():
    return "get_user_input"
return "call_llm"
```

Empty input is never passed to the LLM.

---

# Task 3 : Parallel Execution: Llama + Qwen

Modified graph:

```
get_user_input
    ↓
fanout_models
    ├── call_llama
    └── call_qwen
           ↓
    print_both_responses
           ↓
    get_user_input
```

LangGraph supports parallel execution via multiple outgoing edges. The join node prints both model outputs.

---

# Task 4 : Conditional Model Switching

Parallel execution replaced with conditional routing:
- If input begins with `"Hey Qwen"` → route to Qwen
- Otherwise → route to Llama

Only one model runs per turn.

---

# Task 5 : Adding Chat History (Message API)

Initially, the system was stateless.

Added:

```python
messages: List[Message]
```

Each message:

```json
{"role": "system" | "human" | "ai" | "tool", "content": "..."}
```

Before invoking the model:

```python
prompt = build_prompt_from_messages(messages)
```

The full transcript is passed every turn, simulating a chat model. Conversation state now persists across turns.

---

# Task 6 : Multi-Agent Shared History (Human + Llama + Qwen)

## Problem

The Message API supports only:
- system
- user
- assistant
- tool

But we have three entities:
- Human
- Llama
- Qwen

## Solution

Store canonical history as:

```json
{"speaker": "Human" | "Llama" | "Qwen" | "Tool" | "System"}
```

When building prompts:
- Human → role `"user"` with content `"Human: ..."`
- If previous speaker == target model → role `"assistant"`
- If previous speaker != target model → role `"user"` with speaker prefix

## Example

If **Qwen** is called:

```json
[
  {"role": "system", "content": "...Qwen instructions..."},
  {"role": "user", "content": "Human: What is the best ice cream flavor?"},
  {"role": "user", "content": "Llama: There is no one best flavor..."}
]
```

If **Llama** is called later:

```json
[
  {"role": "system", "content": "...Llama instructions..."},
  {"role": "user", "content": "Human: What is the best ice cream flavor?"},
  {"role": "assistant", "content": "Llama: There is no one best flavor..."},
  {"role": "user", "content": "Qwen: No way, chocolate is the best!"},
  {"role": "user", "content": "Human: I agree."}
]
```

Each model receives a tailored system prompt describing participants.

---

# Task 7 : Checkpointing and Crash Recovery

LangGraph supports durable execution and recovery.

Implemented manual checkpointing using:

```
task7_lg_checkpoint.json
```

Checkpoint stores:

- `messages`
- `verbose`
- `last_model`

Checkpoint saved:
- After each human input
- After each model reply
- On reset
- On clean exit
- On SIGINT / SIGTERM

Atomic writes prevent corruption.

## Restart Behavior

On startup:
- If checkpoint exists → restore state
- If not → initialize new conversation

You can:
- Start conversation
- Kill program mid-dialog
- Restart program
- Continue with full history intact

## What This Demonstrates

This project demonstrates:
- LangGraph node orchestration  
- Conditional routing  
- Parallel execution  
- Dynamic model selection  
- Shared multi-agent context  
- Message API translation  
- Output sanitization  
- Crash-safe persistence  
- Durable agent architecture  

## Example Conversation

```
Human: What is the best ice cream flavor?
Llama: Vanilla is versatile.

Human: Hey Qwen, what do you think?
Qwen: Chocolate is superior.

Human: I agree.
Llama: Both are classics!
```

Kill the program.

Restart.

Conversation resumes with full history.

---

# 🏗 Architectural Evolution

| Stage | Capability |
|-------|------------|
| 1 | Single Llama node |
| 2 | Empty-input routing |
| 3 | Parallel Llama + Qwen |
| 4 | Conditional switching |
| 5 | Chat history memory |
| 6 | Multi-agent shared history |
| 7 | Durable checkpoint recovery |

---

# 🎯 Conclusion

This project evolves a simple LangGraph LLM wrapper into a:

- Multi-agent conversational system  
- Shared-context dialogue framework  
- Dynamically routed architecture  
- Crash-resilient persistent agent  

It demonstrates how **LangGraph** can be used to build production-style, recoverable multi-agent LLM systems.