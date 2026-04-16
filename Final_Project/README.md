# Cognitive Court

> Demo: **Paste your GitHub demo link here** — for example, your deployed Streamlit app, GitHub Pages link, or repository demo URL.

Cognitive Court is a scientific reasoning QA system that compares answers from multiple LLMs, runs a judge model, optionally checks claims with a verifier, and highlights the best response. It includes both a CLI workflow and a Streamlit UI for interactive testing.

## What it does

- Generates answers from multiple models in parallel
- Supports two answer styles:
  - **Standard**: concise final answer only
  - **CoT**: structured scientific reasoning format
- Uses a judge model to score and compare answers
- Optionally runs a verifier to flag unsupported claims
- Resolves ties with a deterministic tie-break system
- Provides a polished Streamlit interface for side-by-side comparison
- Includes a mock mode for local testing without live API calls

## Project structure

```text
.
├── cognitive_court_app.py   # Streamlit app
├── data/
│   └── test_questions.json   # Example scientific QA questions
├── src/
│   ├── clients.py            # OpenAI-compatible client wrappers and mock client
│   ├── config.py             # Environment-based configuration
│   ├── main.py               # CLI entry point
│   ├── pipeline.py           # Answer generation, judging, verification logic
│   └── prompts.py            # Prompt templates
└── outputs/                  # JSONL run outputs
```

## Requirements

- Python 3.10+
- An OpenAI-compatible API for the judge/verifier
- An OpenRouter-compatible API for the answer models
- Streamlit for the UI

## Installation

```bash
git clone https://github.com/your-username/cognitive-court.git
cd cognitive-court
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you do not already have a `requirements.txt`, install the main dependencies manually:

```bash
pip install openai python-dotenv tqdm streamlit
```

## Environment variables

Create a `.env` file in the project root:

```bash
MOCK_MODE=0

OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_API_KEY=your_openrouter_key

OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_openai_key

QWEN_MODEL=qwen/qwen-2.5-7b-instruct
MISTRAL_MODEL=mistralai/ministral-8b-2512
LLAMA_MODEL=meta-llama/llama-3.1-8b-instruct
JUDGE_MODEL=gpt-4o-mini

TEMPERATURE=0.2
ANSWER_MAX_NEW_TOKENS=500
JUDGE_MAX_NEW_TOKENS=900
VERIFIER_MAX_NEW_TOKENS=250
SEED=42
```

## Run the CLI

Generate answers and save a JSONL run:

```bash
python -m src.main run --questions data/test_questions.json --output outputs/run.jsonl
```

Limit the number of questions:

```bash
python -m src.main run --questions data/test_questions.json --output outputs/run.jsonl --limit 5
```

Evaluate a saved JSONL file:

```bash
python -m src.main evaluate --input outputs/run.jsonl
```

### Judge stability checks

Run the judge multiple times with different seeds:

```bash
python -m src.main run --questions data/test_questions.json --output outputs/run.jsonl --judge-runs 5 --judge-seed-step 1
```

## Run the Streamlit app

```bash
streamlit run cognitive_court_app.py
```

The UI lets you:

- paste a scientific question
- choose **Standard**, **CoT**, or **Both**
- toggle claim verification
- switch review style between **Balanced**, **Strict**, and **Exploratory**
- compare model answers side by side

## Mock mode

For local testing without API access, set:

```bash
MOCK_MODE=1
```

Mock mode uses deterministic heuristic clients so you can test the full pipeline, including answer generation, judging, and verification.

## Data format

Question files should be JSON arrays with entries like:

```json
{
  "id": "Q001",
  "question": "Why does increasing atmospheric CO2 lead to higher global temperatures?",
  "category": "climate science",
  "reasoning_type": "causal",
  "difficulty": "easy",
  "gold_explanation": "CO2 absorbs outgoing infrared radiation, trapping more heat in the atmosphere and strengthening the greenhouse effect."
}
```

## Output format

Runs are written as JSONL, one record per question. Each record can include:

- the original question metadata
- gatekeeper decision
- answers from each model
- judge output
- verifier output
- baselines and tie-break information

## Notes

- The judge prompt emphasizes reasoning quality, mechanism coverage, coherence, and depth.
- The verifier scores claims for support, unsupported claims, and uncertainty.
- The Streamlit app highlights the overall best answer across modes.
- `data/test_questions.json` contains 50 example scientific questions across physics, biology, chemistry, medicine, climate science, and more.


