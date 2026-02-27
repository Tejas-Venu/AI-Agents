# Topic5RAG — Complete RAG Experiment Documentation

This repository documents Retrieval-Augmented Generation (RAG) experiments conducted across Exercises 1–11.

Each exercise section below contains:
- Tasks performed
- The exact questions
- The exact answers (verbatim, exactly as provided)

---

# Table of Contents

1. [Project Structure](#project-structure)  
2. [Exercise 1 — Open Model RAG vs No RAG](#exercise-1--open-model-rag-vs-no-rag)  
3. [Exercise 2 — Open Model RAG vs Large Model](#exercise-2--open-model--rag-vs-large-model-comparison)   
4. [Exercise 3 — Open Model + RAG vs. State-of-the-Art Chat Model](#exercise-3--open-model--rag-vs-state-of-the-art-chat-model)  
5. [Exercise 4 — Effect of Top-K Retrieval Count](#exercise-4--effect-of-top-k-retrieval-count)  
6. [Exercise 5 — Handling Unanswerable Questions](#exercise-5--handling-unanswerable-questions)  
7. [Exercise 6 — Query Phrasing Sensitivity](#exercise-6--query-phrasing-sensitivity)  
8. [Exercise 7 — Chunk Overlap Experiment](#exercise-7--chunk-overlap-experiment)  
9. [Exercise 8 — Chunk Size Experiment](#exercise-8--chunk-size-experiment)  
10. [Exercise 9 — Retrieval Score Analysis](#exercise-9--retrieval-score-analysis)  
11. [Exercise 10 — Prompt Template Variations](#exercise-10--prompt-template-variations)  
12. [Exercise 11 — Cross-Document Synthesis](#exercise-11--cross-document-synthesis)  

---

# Project Structure

```
Topic5RAG/
├── Exercise 1/
│   ├── Exercise1.ipynb
│   ├── observation.txt
|   ├──  Congressional Record/
│   │   ├── output1.txt
│   │   ├── output2.txt
│   │   ├── output3.txt
│   │   └── output4.txt
│   └── Model T Ford/
│       ├── output1.txt
│       ├── output2.txt
│       ├── output3.txt
│       └── output4.txt
│
├── Exercise 2/
│   ├── Exercise2.ipynb
│   ├── observation.txt
│   └── output.txt
│
├── Exercise 3/
│   ├── Exercise3.ipynb
│   ├── observation.txt
│   └── output.txt
│
├── Exercise 4/
│   ├── Exercise4.ipynb
│   ├── observation.txt
│   └── output.txt
│
├── Exercise 5/
│   ├── Exercise5.ipynb
│   ├── modified_prompt_observation.txt
│   ├── modified_prompt_output.txt
│   ├── original_prompt_observation.txt
│   └── original_prompt_output.txt
│
├── Exercise 6/
│   ├── Exercise6.ipynb
│   ├── observations.txt
│   └── output.txt
│
├── Exercise 7/
│   ├── Exercise7.ipynb
│   ├── observations.txt
│   └── output.txt
│
├── Exercise 8/
│   ├── Exercise8.ipynb
│   ├── observation.txt
│   └── output.txt
│
├── Exercise 9/
│   ├── Exercise9.ipynb
│   ├── observation.txt
│   ├── with_threshold_output.txt
│   └── without_threshold_output.txt
│
├── Exercise 10/
│   ├── Exercise10.ipynb
│   ├── observation.txt
│   └── output.txt
│
├── Exercise 11/
│   ├── Exercise11.ipynb
│   ├── observation.txt
│   └── output.txt
│
└── README.md
```

---

# Exercise 1 — Open Model RAG vs. No RAG Comparison

## Tasks Performed

- Use Qwen 2.5 1.5B (or another small open model) with the Model T Ford repair manual, and then with the Congressional Record corpus (separately).
- Ask the model directly (no RAG).
- Ask using your RAG pipeline.
- Compare hallucination vs grounding.

### Question:
Does the model hallucinate specific values without RAG?

Answer:
Yes, the model hallucinates specific values and detailed factual claims without RAG. It invents dates, dollar amounts, legislative details, and events while presenting them with high confidence.

### Question:
Does RAG ground the answers in the actual manual?

Answer:
Yes, RAG helps ground the responses in retrieved context. The answers become more aligned with documented material and contain fewer fabricated specifics.

### Question:
Are there questions where the model's general knowledge is actually correct?

Answer:
In these examples, there are no clear cases where the model’s general knowledge alone produced a reliably correct answer. While some dates may coincidentally match those mentioned in the questions, the surrounding details — such as legislative actions, dollar amounts, and event descriptions — are fabricated or unsupported. Therefore, even if a date appears correct, the overall response cannot be considered accurate.

---

# Exercise 2 — Open Model + RAG vs. Large Model Comparison

## Tasks Performed

- Write a program to run GPT 4o Mini with no tools on individual questions.
- Run it on the queries from Exercise 1.
- Compare hallucination behavior and correctness.

### Question:
Does GPT-4o Mini do a better job than Qwen 2.5 1.5B in avoiding hallucinations?

Answer:
Yes. GPT-4o Mini avoids fabricating specific events in the 2026 questions and instead clearly states its knowledge cutoff (October 2023), whereas the smaller Qwen 2.5 1.5B model previously generated detailed but incorrect claims and invented facts. While GPT-4o Mini still gives a likely incorrect description of the Main Street Parity Act, it is generally more cautious and significantly less prone to confident hallucination about unknown future events.

### Question:
Which questions does GPT-4o Mini answer correctly? Compare the cut-off date of GPT-4o Mini pre-training and the age of the Model T Ford and Congressional Record corpora.

Answer:
GPT-4o Mini correctly handles the 2026 questions about Mr. Flood and Elise Stefanovic by refusing to fabricate post-2023 events, which aligns with its October 2023 knowledge cutoff. Its answer about congressional debate over pregnancy center funding is broadly plausible but generic, while its explanation of the Main Street Parity Act appears incorrect. Since GPT-4o Mini’s training cutoff is October 2023, it predates 2026 Congressional Record content but postdates historical materials like the Model T Ford era (1908–1927) and most of the long-standing Congressional Record archive, meaning it could have been trained on historical congressional data but not future proceedings.

---

# Exercise 3 — Open Model + RAG vs. State-of-the-Art Chat Model

## Tasks Performed

- Local: Qwen 2.5 1.5B with RAG using the Model T manual.
- Cloud: GPT-4 or Claude via web interface (no file upload).
- Run all Exercise 1 queries.
- Compare performance.

### Question:
Where does the frontier model's general knowledge succeed?

Answer:
The frontier model’s general knowledge succeeds on broadly documented, historical, or mechanical questions such as Model T maintenance (carburetor adjustment, spark plug gap, transmission band tightening, oil type). These are well-established topics that are likely covered in training data and do not require access to a specific proprietary document. It also appropriately declined to fabricate details about future 2026 congressional events.

### Question:
When did the frontier model appear to be using live web search to help answer your questions?

Answer:
There is no clear indication that live web search was used. The responses to the 2026 congressional questions explicitly referenced a knowledge cutoff and declined to provide post-2023 information, which suggests the model relied solely on pretraining rather than real-time retrieval.

### Question:
Where does your RAG system provide more accurate, specific answers?

Answer:
The RAG system provides more accurate and specific answers when questions require precise details from a particular source, such as the Model T manual or exact Congressional Record entries. In those cases, retrieval grounds the answer in the actual text, reducing hallucination and increasing factual precision.

### Question:
What does this tell you about when RAG adds value vs. when a powerful model suffices?

Answer:
This comparison shows that powerful frontier models are often sufficient for general knowledge and widely documented topics. However, RAG adds significant value when answering domain-specific, document-specific, or time-sensitive questions that are not reliably stored in pretraining data. In short, general knowledge favors large models, while specialized or corpus-specific tasks benefit strongly from RAG.

---

# Exercise 4 — Effect of Top-K Retrieval Count

## Tasks Performed

- Test with k = 1, 3, 5, 10, 20.
- Run the same queries.
- Observe quality and latency.

### Question:
At what point does adding more context stop helping?

Answer:
Adding more context stops helping once the retrieved passages no longer contain new, relevant information for answering the question. In the examples above, performance improves from very low k (e.g., TOP_K=1) to moderate k (around 5–10), where the correct details begin to appear. Beyond that point (e.g., TOP_K=20), additional context mostly repeats or dilutes useful information rather than improving accuracy.

### Question:
When does too much context hurt (irrelevant information, confusion)?

Answer:
Too much context hurts when irrelevant passages are included, causing the model to latch onto incorrect details or mix unrelated sections (such as rear axle oil being mistaken for engine oil). Larger k values introduced confusion, incorrect numbers (e.g., spark plug gap misreadings), and blended instructions from different sections. This demonstrates that excessive or noisy context can degrade answer quality by overwhelming the model with competing signals.

### Question:
How does k interact with chunk size?

Answer:
k and chunk size are tightly coupled: smaller chunks often require a higher k to capture enough relevant information, while larger chunks may need a smaller k because each chunk already contains more context. If chunks are too large and k is high, the model receives excessive and partially irrelevant information, increasing confusion. Effective RAG performance requires balancing chunk size and k so that the model receives sufficient but focused context.

---

# Exercise 5 — Handling Unanswerable Questions

## Tasks Performed

- Test off-topic, missing, and false-premise questions.
- Modify prompt with refusal instruction.
- Compare hallucination behavior.

### Question:
Does the model admit it doesn't know?

Answer:
Yes, in several RAG cases the model correctly admits that the corpus does not contain the requested information (e.g., horsepower and President questions), but without RAG it often answers confidently from general knowledge even when the answer may be outdated.

### Question:
Does it hallucinate plausible-sounding but wrong answers?

Answer:
Yes, the model hallucinates detailed but incorrect information, such as claiming the 1925 Model T had a six-cylinder engine and recommending synthetic oil in a historical manual where that would be unlikely.

### Question:
Does retrieved context help or hurt? (Does irrelevant context encourage hallucination?)

Answer:
Retrieved context helps when the question is corpus-specific, but irrelevant context can hurt by encouraging the model to fabricate justifications or force connections (e.g., inferring oil recommendations or explaining France using unrelated manual text).

### Question:
Does modified prompt help?

Answer:
With the modified prompt, the model correctly refuses to answer when the corpus does not contain the relevant information, instead stating, “I cannot answer this from the available documents.” This significantly reduces hallucination and prevents the model from fabricating answers based on general knowledge when operating in RAG mode.The contrast is clear: without RAG, the model answers from pretraining (sometimes incorrectly or with outdated details), but with the stricter prompt in RAG mode, it properly respects document boundaries and avoids inventing unsupported facts.

---

# Exercise 6 — Query Phrasing Sensitivity

## Tasks Performed

- Rephrase one underlying question 5+ ways.
- Record top 5 retrieved chunks.
- Compare similarity and overlap.

### Question:
Which phrasings retrieve the best chunks?

Answer:
The most formal and technically precise phrasings — such as “What is the specified spark plug gap…”, “What spark plug electrode gap measurement is recommended…” and “What is the proper spacing…” — retrieve the most consistent and overlapping top chunks (Jaccard = 1.0 between several of them). The maintenance-focused phrasing also performs strongly, while the casual phrasing retrieves slightly more varied results.

### Question:
Do keyword-style queries work better or worse than natural questions?

Answer:
The keyword-style query (“Model T spark plug gap specification”) performs reasonably well but shows lower overlap (0.5 with some formal versions) compared to the most structured natural questions. This suggests keyword queries are effective but slightly less stable than well-formed, semantically clear natural language questions.

### Question:
What does this tell you about potential query rewriting strategies?

Answer:
This suggests that semantically precise, technically worded queries produce the most stable retrieval, so query rewriting strategies should favor clarity, inclusion of domain-specific terms (e.g., “electrode,” “measurement,” “specification”), and removal of ambiguity. Automated rewriting that normalizes casual phrasing into structured, technical language could improve retrieval consistency.

---

# Exercise 7 — Chunk Overlap Experiment

## Tasks Performed

- Re-chunk corpus with overlap = 0, 64, 128, 256.
- Rebuild index.
- Test boundary-spanning questions.

### Question:
Does higher overlap improve retrieval of complete information?

Answer:
Yes — the outputs show low overlaps (0–128) returned partial fragments while overlap=256 returned a chunk containing the spark-plug spec, so higher overlap did recover the complete info. In other cases (carburetor, transmission band, oil) higher overlap improved the quality and completeness of retrieved passages.

### Question:
What's the cost? (Index size, redundant information in context)

Answer:
Cost is large: index size (number of chunks) more than doubled from 449 (overlap 0) to 1017 (overlap 256), meaning more storage and longer retrieval times. It also increases redundancy in the retrieved context (more repeated text across top-K), which inflates prompt length and may waste model budget.

### Question:
Is there a point of diminishing returns?

Answer:
Yes, benefits taper: moderate overlap (64–128) gave noticeable improvements over 0, but moving from 128 → 256 produced smaller gains while dramatically increasing index size. Practically, 64–128 is often the sweet spot; beyond that you pay heavy costs for limited extra recovery.

---

# Exercise 8 — Chunk Size Experiment

## Tasks Performed

- Chunk corpus at 128, 512, 2048 characters.
- Rebuild index.
- Run same queries.

### Question:
How does chunk size affect retrieval precision (relevant vs. irrelevant content)?

Answer:
Smaller chunks (128) give many narrow hits with high topical precision but often miss surrounding context, while very large chunks (2048) return fewer results that *contain* the answer but also include lots of irrelevant nearby text. Medium chunks (512) strike a balance — reasonably focused excerpts with less noisy filler than the largest chunks.

### Question:
How does it affect answer completeness?

Answer:
Larger chunks are far more likely to contain a complete answer inside a single chunk (so overlap/assembly isn’t needed), whereas small chunks frequently split answers across multiple hits and require composing top-K pieces to get a full response. Medium chunks often achieve good completeness without excessive composition.

### Question:
Is there a sweet spot for your corpus?

Answer:
Yes, for this Model T corpus with sentence-aware chunking and overlap=128, ~512 characters appears to be the sweet spot: good precision, manageable index size, and high chance the top results contain usable answer material. Going to 2048 improved single-chunk completeness but at the cost of index size and more irrelevant context.

### Question:
Does optimal size depend on the type of question?

Answer:
Yes, short factual lookups (dates, numeric specs) can work well with smaller chunks, while procedural or descriptive questions (how-to steps, diagnostics) benefit from larger chunks that preserve full instructions and context. We have to choose chunk size based on whether we need concise facts (smaller) or coherent multi-sentence guidance (medium→large).

---

# Exercise 9 — Retrieval Score Analysis

## Tasks Performed

- Retrieve top 10 chunks for 10 queries.
- Record similarity scores.
- Test thresholding.

### Question:
When is there a clear "winner"?

Answer:
A clear winner appears when the top score is noticeably higher than the rest (e.g., How do I tighten the brake and reverse bands?— 0.5662 vs next 0.4140), indicating a high-confidence, focused match. In practice those cases have top scores ≳0.55 and a gap ≳0.12.

### Question:
When are scores tightly clustered (ambiguous)?

Answer:
Scores are tightly clustered (e.g., carburetor, spark-plug, oil, generator queries with many scores around 0.32–0.50) when multiple chunks are similarly relevant or OCR/noise reduces discriminative signal, making retrieval ambiguous. Those clusters indicate the model cannot strongly prefer a single chunk.

### Question:
What score threshold would you use to filter out irrelevant results?

Answer:
Based on these distributions, 0.50 is a reasonable high-precision cutoff (keeps only very confident hits), while ~0.40–0.45 is a better tradeoff for preserving recall but filtering obvious noise. Use 0.5 for precision-critical tasks and 0.4–0.45 when you need more coverage.

### Question:
How does score distribution correlate with answer quality?

Answer:
Higher top scores (and larger top-1/top-2 gaps) generally correlate with more focused, answerable retrieval (better answer quality), while low/flat score distributions correlate with partial, noisy, or incomplete answers. In other words, strong, isolated peaks → better answers; flat profiles → likely partial/incomplete results.

### Question:
How does threshold affect results?

Answer:
Applying a 0.5 threshold dramatically reduces returned results (only 2 of 10 queries returned any chunk), which raises precision but at the cost of recall — many plausible answers are dropped because their top scores fell below 0.5. So thresholding makes the system conservative: fewer, higher-confidence hits but more missed answers that might still be useful.

---

# Exercise 10 — Prompt Template Variations

## Tasks Performed

- Tested minimal, strict grounding, citation, permissive, and structured prompts.
- Evaluated accuracy and helpfulness.

### Question:
Which prompt produces the most accurate answers?

Answer:
Citation tends to produce the most verifiable answers because it forces quoting and source tags, so when the retrieved text is clean it yields the closest match to the manual. However, OCR/noise in your corpus (e.g. the garbled “¥;” spark-gap) can make even citation answers look wrong, so accuracy depends on retrieval/cleanliness too.

### Question:
Which produces the most useful answers?

Answer:
Structured and permissive are the most useful: structured gives organized facts + a concise synthesis that’s easy to act on, while permissive produces fuller, practitioner-oriented answers. Between them, structured is safer for traceability and permissive is slightly more practical when the corpus is incomplete.

### Question:
Is there a trade-off between strict grounding and helpfulness?

Answer:
Yes, enforcing strict grounding or citation increases traceability and reduces hallucination risk but can make responses shorter, more cautious, or fail when source text is noisy. Conversely, permissive prompts boost helpful, actionable text (higher apparent usefulness) at the cost of potentially adding unsupported or paraphrased content.

---

# Exercise 11 — Cross-Document Synthesis

## Tasks Performed

- Designed synthesis queries combining multiple chunks.
- Tested k = 3, 5, 10.
- Evaluated completeness and contradictions.

### Question:
Does retrieving more chunks improve synthesis?

Answer:
Retrieving more chunks increases the amount of information available and leads to longer, more comprehensive answers. However, in your results it did not improve measurable groundedness and sometimes introduced extra or loosely supported details.

### Question:
Can the model successfully combine information from multiple chunks?

Answer:
Yes, the model is able to merge information from different retrieved chunks into a single synthesized response. However, it often paraphrases heavily and may blend details in ways that reduce traceable grounding.

### Question:
Does it miss information that wasn't retrieved?

Answer:
Yes, the model generally cannot include information that was not retrieved in the top-k results. Retrieval acts as a bottleneck, so missing chunks directly lead to missing content in the answer.

### Question:
Does contradictory information in different chunks cause problems?

Answer:
Yes, contradictory or slightly different instructions across chunks can lead to vague, blended, or inconsistent answers. The model typically does not explicitly resolve contradictions unless prompted to compare or cite sources carefully.

---