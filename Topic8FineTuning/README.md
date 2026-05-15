# Topic 8 - Fine-Tuning Llama for Text-to-SQL Generation

This project implements and evaluates:

- Fine-tuning `meta-llama/Llama-3.2-1B` using Tinker LoRA training  
- Prompt engineering for Text-to-SQL generation  
- Execution-based SQL evaluation using SQLite  
- Dataset preprocessing and loss masking  
- Base model versus fine-tuned model comparison  
- Generalization to unseen database schemas  
- Error analysis of generated SQL queries  
- Comparison between Fine-Tuning and RAG approaches  

---

## Table of Contents

1. [Project Directory](#project-directory)  
2. [Step 0 — Environment Setup](#step-0--environment-setup)  
3. [Step 1 — Load and Explore the Dataset](#step-1--load-and-explore-the-dataset)  
   - [Dataset Structure](#dataset-structure)  
   - [Train/Test Split](#traintest-split)  
4. [Step 2 — Prompt Formatting for Text-to-SQL](#step-2--prompt-formatting-for-text-to-sql)  
5. [Step 3 — Evaluating the Base Model](#step-3--evaluating-the-base-model)  
   - [Execution-Based Evaluation](#execution-based-evaluation)  
6. [Step 4 — Preparing Training Data](#step-4--preparing-training-data)  
   - [Loss Masking Strategy](#loss-masking-strategy)  
7. [Step 5 — Fine-Tuning with LoRA](#step-5--fine-tuning-with-lora)  
8. [Step 6 — Evaluating the Fine-Tuned Model](#step-6--evaluating-the-fine-tuned-model)  
9. [Step 7 — Testing on Novel Schemas](#step-7--testing-on-novel-schemas)  
10. [Discussion Questions](#discussion-questions)  
    - [Before vs After Fine-Tuning](#before-vs-after-fine-tuning)  
    - [RAG Comparison](#rag-comparison)  
    - [Error Analysis](#error-analysis)  
11. [Conclusion](#conclusion)  

---

# Project Directory

```text
Topic8FineTuning/
├── README.md
├── finetuning.ipynb
└── sql_create_context_v4.json
```

---

# Step 0 — Environment Setup

The project uses:

- Tinker
- Transformers
- Python Dotenv
- SQLite-based SQL evaluation

The environment was configured by:

- Installing required dependencies
- Setting the `TINKER_API_KEY`
- Placing the dataset file `sql_create_context_v4.json` in the working directory

The dataset contains:
- Database schemas
- Natural language questions
- Ground-truth SQL queries

---

# Step 1 — Load and Explore the Dataset

The dataset was loaded from JSON format and explored to understand:
- Schema complexity
- Question diversity
- SQL query patterns

The dataset includes:
- Simple SELECT queries
- WHERE clauses
- Aggregations
- GROUP BY
- ORDER BY
- JOINs
- Nested queries

---

## Dataset Structure

Each example contains:

| Field | Description |
|-------|-------------|
| context | Database schema |
| question | Natural language question |
| answer | Ground-truth SQL query |

Example tasks range from:
- Simple filtering
- Aggregation queries
- Multi-table joins
- Complex relational reasoning

---

## Train/Test Split

The dataset was shuffled and divided into:

- 200 held-out test examples
- Remaining examples used for training

This ensured:
- Fair evaluation
- Separation between training and testing data

---

# Step 2 — Prompt Formatting for Text-to-SQL

A consistent prompt template was used:

```text
Table schema:
<schema>

Question:
<question>

SQL:
```

This format teaches the model:
- Schema grounding
- SQL syntax generation
- Question-to-query mapping

The model learns to generate the SQL completion after the `SQL:` token.

---

# Step 3 — Evaluating the Base Model

The base model used:

- `meta-llama/Llama-3.2-1B`

Evaluation was performed before fine-tuning to establish a baseline.

---

## Execution-Based Evaluation

Instead of comparing SQL strings directly, evaluation used:
- SQLite execution
- Result-set comparison

This avoids false negatives when:
- Different SQL syntax produces identical outputs
- Equivalent queries are written differently

The helper utility:
- Builds temporary SQLite databases
- Executes both generated and expected queries
- Compares returned result sets

This provides more reliable accuracy measurement.

---

# Step 4 — Preparing Training Data

Each example was converted into:
- Prompt tokens
- Completion tokens

Only the SQL completion contributes to training loss.

---

## Loss Masking Strategy

The training setup used:
- Zero loss weight for prompt tokens
- Full loss weight for SQL completion tokens

This ensures the model learns:
- SQL generation
- Not prompt reconstruction

Training examples were then:
- Tokenized
- Converted into autoregressive next-token prediction format
- Shuffled before training

---

# Step 5 — Fine-Tuning with LoRA

The project used:
- LoRA fine-tuning
- Adam optimizer
- Cross-entropy loss

Training configuration:
- 1 epoch
- Batch size of 256
- Learning rate of `5e-4`

Observed behavior during training:
- Loss steadily decreased
- SQL syntax improved
- Schema grounding improved

Typical runtime:
- Approximately 10–20 minutes

---

# Step 6 — Evaluating the Fine-Tuned Model

After training:
- LoRA weights were saved
- A new sampling client was created
- Evaluation was repeated on the same 200 held-out test questions

---

## Accuracy Results

| Model | Accuracy |
|-------|-----------|
| Base Model | ~37% |
| Fine-Tuned Model | ~87% |

The fine-tuned model showed major improvements in:
- SQL syntax generation
- Aggregation handling
- JOIN generation
- Schema grounding
- Query structure prediction

---

# Step 7 — Testing on Novel Schemas

Additional evaluation was performed on schemas not present in training.

The tests included:

## Easy Queries
- Simple filtering
- COUNT queries
- Basic WHERE conditions

## Medium Queries
- Aggregations
- ORDER BY
- GROUP BY

## Hard Queries
- JOIN operations
- Multi-table reasoning
- Department-level aggregations

Observed behavior:
- Strong performance on easy and medium tasks
- Reduced performance on hard JOIN-heavy queries
- Occasional mistakes on unseen schema naming conventions

This demonstrates:
- Good in-distribution learning
- Partial generalization to unseen schemas

---

# Discussion Questions

# Before vs After Fine-Tuning

The fine-tuned model demonstrated significant improvements compared to the base model.

## Improvements Observed

The model learned both:
- SQL syntax generation
- Schema grounding

Specific improvements included:
- Better SELECT/FROM/WHERE structure
- Better aggregation handling
- Improved JOIN generation
- Reduced malformed SQL outputs

## Accuracy Change

| Model | Accuracy |
|-------|-----------|
| Base Model | 42% |
| Fine-Tuned Model | 92.50% |

The model generalized well on:
- Filtering queries
- Aggregations
- GROUP BY queries

Performance dropped somewhat on:
- Novel schemas
- Complex JOIN logic
- Multi-hop relational reasoning

---

## Performance on Manual Questions

| Difficulty | Performance |
|------------|-------------|
| Easy | Very Good |
| Medium | Good |
| Hard | Moderate |

Observed behavior:
- Easy schemas were handled correctly most of the time
- Medium tasks occasionally produced ordering/grouping mistakes
- Hard JOIN queries sometimes used incorrect join keys or missed GROUP BY clauses

---

# RAG Comparison

A Retrieval-Augmented Generation (RAG) system with 1,000 `(question, SQL)` pairs would work well for:

- Similar schemas
- Frequently repeated SQL structures
- Simple filtering queries
- Standard aggregations

Examples:
- COUNT queries
- MAX/MIN aggregations
- Basic WHERE filtering

RAG would struggle with:
- Unseen schemas
- Novel table relationships
- Complex JOIN logic
- Relational reasoning

Why?

Because retrieval depends heavily on:
- Semantic similarity
- Schema overlap

If retrieved examples do not closely match the new schema, the model may:
- Use incorrect column names
- Produce wrong joins
- Generate partially mismatched SQL

Fine-tuning improves:
- Internal SQL understanding
- Schema interpretation
- Relational reasoning

instead of relying purely on retrieval.

---

# Error Analysis

When the fine-tuned model failed, the errors generally fell into three categories.

---

## 1. Wrong Column Names

Examples:
- Using semantically similar but nonexistent columns
- Incorrect schema references

This suggests:
- Partial schema grounding
- Semantic understanding without exact schema matching

---

## 2. Wrong SQL Logic

Examples:
- Missing GROUP BY clauses
- Incorrect JOIN conditions
- Wrong aggregation logic

This indicates:
- Weak compositional reasoning
- Incomplete relational planning

---

## 3. SQL Syntax Errors

Examples:
- Missing commas
- Incorrect aliases
- Malformed ORDER BY clauses

These errors became significantly less common after fine-tuning.

---

# Conclusion

This project demonstrates:

- Effective LoRA fine-tuning for Text-to-SQL generation  
- Major improvements from supervised fine-tuning  
- Strong gains in SQL syntax generation and schema grounding  
- Execution-based SQL evaluation using SQLite  
- Generalization testing on unseen schemas  
- Real-world limitations of small language models on complex relational reasoning  

The final system integrates:

- Prompt engineering  
- Execution-based evaluation  
- Parameter-efficient fine-tuning  
- Structured SQL reasoning  
- Error analysis and benchmarking  

within a lightweight and reproducible Text-to-SQL pipeline.