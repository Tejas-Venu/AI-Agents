# Topic 7 - MCP

---

## Table of Contents

1. [Project Directory](#project-directory)  
2. [Exercise A — Discover the Asta Tools](#exercise-a--discover-the-asta-tools)  
3. [Exercise B — Direct Asta Tool Calls](#exercise-b--direct-asta-tool-calls)  
4. [Exercise C — Asta-Powered Research Chatbot](#exercise-c--asta-powered-research-chatbot)  
5. [Exercise D — Citation Network Explorer Agent](#exercise-d--citation-network-explorer-agent)  
6. [Closing Discussion](#closing-discussion)  
7. [Conclusion](#conclusion)  

---

# Project Directory

```text
Topic7MCP/
├── A2A
└── MCP/
    ├── exerciseA.py
    ├── exerciseB.py
    ├── exerciseC.py
    ├── exerciseD.py
    ├── output/
    |    ├── exerciseA.txt
    │    ├── exerciseB.txt
    │    ├── exerciseC.txt
    │    └── exerciseD.txt
    └── README.md
```

---

# Exercise A — Discover the Asta Tools

`exerciseA.py` connects directly to the Asta MCP endpoint and retrieves all available tool schemas using the `tools/list` MCP method.

## Features Implemented

- Sends a JSON-RPC `tools/list` request
- Handles both:
  - `application/json`
  - `text/event-stream` (SSE)
- Extracts:
  - tool names
  - descriptions
  - required parameters
  - optional parameters
- Dynamically parses JSON Schema definitions
- Handles nested schemas and arrays safely
- Gracefully handles malformed or unexpected responses

---

# Exercise B — Direct Asta Tool Calls

`exerciseB.py` performs three direct MCP tool-calling drills without involving an LLM.

The script focuses on:

- robust SSE handling
- JSON extraction
- noisy response filtering
- deduplication of parsed papers

---

## Drill 1 — Search Papers

### Query

```text
large language model agents
```

### Tool Used

```text
search_papers_by_relevance
```

### Requested Fields

```text
title,abstract,year,authors
```

### Output

The script prints:

- top 5 paper titles
- publication years

### Implementation Details

The helper:

```python
collect_papers_from_outputs(outputs)
```

extracts paper-like objects from:

- dictionaries
- lists
- raw text blobs

and deduplicates them by normalized title.

---

## Drill 2 — Get Citations

### Target Paper

```text
ARXIV:1810.04805
```

(BERT paper)

### Tool Used

```text
get_citations
```

### Filter

```text
publication_date_range = "2023-01-01:"
```

### Output

The script prints:

- total parsed citation count
- titles of the first 5 citing papers

### Implementation Details

The parser handles multiple response structures such as:

- `citingPaper`
- `citations`
- `results`
- `data`

It also extracts title-like lines from noisy text responses.

---

## Drill 3 — Get References

### Target Paper

```text
ARXIV:2210.03629
```

(ReAct paper)

### Tool Used

```text
get_references
```

### Output

The script prints:

- reference titles
- reference years
- sorted ascending by year

### Implementation Details

The script:

- removes noisy non-paper text
- deduplicates references
- sorts safely using regex-based year extraction

---

# Exercise C — Asta-Powered Research Chatbot

`exerciseC.py` implements a research chatbot powered by:

- GPT-4o mini
- MCP tool discovery
- dynamic function calling

The chatbot automatically discovers all available Asta tools at startup and converts them into OpenAI-compatible function schemas.

---

## Example Queries

### Search papers

```text
Find recent papers about large language model agents
```

### Citation analysis

```text
What papers cite the original BERT paper?
```

### Multi-tool chaining

```text
Who wrote Attention is All You Need and what else have they published?
```

### Reference summarization

```text
Summarize the references used in the ReAct paper
```

---

# Exercise D — Citation Network Explorer Agent

`exerciseD.py` implements an autonomous citation analysis agent.

Unlike the chatbot, the agent controls the tool-calling order directly instead of letting the LLM decide.

The LLM is used only for generating the final markdown report.

---

## Features Implemented

- Seed paper retrieval
- Citation neighborhood construction
- Reference analysis
- Recent citation analysis
- Author profile analysis
- Markdown report generation
- Topic-based seed paper discovery
- Recurring collaboration detection
- Research gap identification

---

## Agent Workflow

### Step 1 — Retrieve Seed Paper

The agent retrieves:

- title
- abstract
- year
- authors
- references

using:

```text
get_paper
```

---

### Step 2 — Analyze References

The agent:

- extracts reference IDs
- retrieves metadata
- ranks references by citation count
- selects the top 5 foundational works

---

### Step 3 — Retrieve Recent Citations

The agent retrieves papers citing the seed paper from the last 3 years using:

```text
get_citations
```

and selects the 5 most recent citing papers.

---

### Step 4 — Analyze Authors

For each author:

- retrieve author papers
- identify the most-cited non-seed paper
- build an author profile

using:

```text
get_author_papers
```

---

### Step 5 — Detect Recurring Collaborations

The agent compares:

- reference authors
- citing-paper authors

and detects overlapping names.

---

### Step 6 — Generate Final Report

The structured data is compressed and sent to GPT-4o mini to generate a markdown report containing:

- seed paper summary
- foundational works
- recent developments
- author profiles
- recurring collaborations
- research gaps

---

## Example Usage

### Using a paper ID

```bash
python citation_explorer.py ARXIV:2210.03629
```

### Using a topic keyword

```bash
python citation_explorer.py topic:large language model agents
```

---

# Closing Discussion

## What does MCP automation buy you? What does it cost?

MCP removes the need to manually write tool schemas and allows clients to dynamically discover new capabilities from a server. The tradeoff is increased complexity in parsing responses, handling transport protocols like SSE, and dealing with inconsistent response formats.

## How did you decide what to include in the context window?

Only the most relevant fields such as titles, abstracts, years, and citation counts were retained because passing entire raw responses would quickly increase token usage. Summarized and filtered data produced cleaner prompts and more focused final reports.


## What would it take to let the LLM decide tool order in Exercise D?

The system would need a planning layer that allows the model to reason about dependencies between tool calls. Without careful control, the model could make redundant calls, request missing information, or generate inefficient execution chains.


## What would a mature MCP ecosystem provide?

A mature ecosystem would likely provide stronger schema consistency, standardized authentication flows, tool registries, improved discovery mechanisms, and better interoperability between MCP servers and clients. More reliable streaming standards and validation tooling would also reduce integration complexity.

---

# Conclusion

The final system integrates:
- direct MCP communication
- dynamic OpenAI function calling
- structured citation analysis
- autonomous report generation
within a flexible and extensible research-agent framework.