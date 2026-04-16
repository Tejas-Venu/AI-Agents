from __future__ import annotations
import json
from typing import Dict


STANDARD_ANSWER_PROMPT = """\
You are answering a scientific question.

Return only the final answer.

Rules:
- Do not show reasoning, thinking process, analysis, or intermediate steps.
- Do not use headings, numbering, or bullet points.
- Do not mention that you are thinking.
- Keep it concise and direct.
"""


def build_standard_prompt(question: str) -> str:
    return f"""{STANDARD_ANSWER_PROMPT}

Question:
{question}
"""


COT_ANSWER_PROMPT = """\
You are answering a scientific reasoning question.

Here is an example of how to answer:

Question:
Why does increasing temperature increase reaction rates?

Answer:
1. Main answer:
Increasing temperature increases reaction rates because particles move faster and collide more frequently with enough energy.

2. Explanation:
When temperature rises, molecules gain kinetic energy. This leads to more frequent collisions and increases the likelihood that collisions have enough energy to overcome activation energy, resulting in more successful reactions.

3. Key reasoning steps:
- Step 1: Higher temperature increases molecular motion.
- Step 2: Faster molecules collide more often.
- Step 3: More collisions exceed activation energy.
- Step 4: Reaction rate increases.

Now answer the following question in the same format.

Use this EXACT structure:

1. Main answer:
2. Explanation:
3. Key reasoning steps:
- Step 1:
- Step 2:
- Step 3:
"""


def build_cot_prompt(question: str) -> str:
    return f"""{COT_ANSWER_PROMPT}

Question:
{question}
"""


# Optional backward-compatible alias
def build_answer_prompt(question: str) -> str:
    return build_cot_prompt(question)


JUDGE_PROMPT = """\
You are an expert evaluator of scientific reasoning.

Your job is to evaluate multiple answers to a scientific question and choose the best one based on the QUALITY OF REASONING.

IMPORTANT:
- The answers are anonymized labels such as A, B, C, ...
- Evaluate reasoning quality, NOT length or formatting.
- Do NOT reward verbosity, repetition, or template structure.
- A short answer can win ONLY if it explains the mechanism completely and does not leave out any key details.
- A long answer should win ONLY if it adds NEW, correct, and relevant scientific information.
- Be strict: missing steps in reasoning should be penalized.

--------------------------------------------------
DEFINITIONS (STRICT SCORING RUBRIC)
--------------------------------------------------

correctness:
- 0 = incorrect or misleading
- 1 = partially correct
- 2 = fully correct

mechanism_coverage:
Count DISTINCT causal/mechanistic steps in the explanation.
- 0 = no mechanism
- 1 = single-step explanation
- 2 = two-step causal chain
- 3 = multi-step mechanism with a clear chain of 3 or more distinct steps

logical_coherence:
- 0 = disorganized, contradictory, or unclear
- 1 = somewhat logical but incomplete or loosely connected
- 2 = clear, step-by-step progression with explicit causal links

explanatory_depth:
Measure how much NEW scientific explanation the answer adds beyond the core answer.
Depth is about distinct mechanistic detail, not length.

- 0 = restates the answer only, or gives no real explanation
- 1 = gives a minimal explanation with little causal detail
- 2 = adds meaningful mechanistic detail, such as intermediate steps or conditions
- 3 = adds deeper insight, such as:
  - intermediate states
  - limiting factors
  - conditions under which the mechanism changes
  - comparisons or interactions between multiple processes
  - why the mechanism leads to the outcome, not just that it does

Important:
- Repeating the same idea in different words does NOT increase depth.
- Extra sentences do NOT increase depth unless they add new scientific content.
- A detailed but repetitive answer should score lower than a shorter but more informative answer.

unsupported_claims_penalty:
- 0 = no unsupported claims
- 1 = minor speculation, vague claim, or slightly shaky statement
- 2 = significant unsupported, misleading, or incorrect claims

CRITICAL RULES:
- Repeating the same idea DOES NOT increase depth.
- Step-by-step formatting DOES NOT increase score unless each step adds NEW information.
- Prefer answers that explain WHY and HOW, not just WHAT.
- Penalize missing intermediate steps in causal chains.
- Penalize answers that sound detailed but do not add new scientific content.

--------------------------------------------------
EVALUATION PROCEDURE
--------------------------------------------------

Step 1: Identify question type
- causal / mechanism / counterfactual

Step 2: For EACH answer:
- Break the answer into 2-5 atomic claims (short factual statements).
- Evaluate using the rubric above.

Step 3: Choose the winner:
- Select the answer with the best overall scientific reasoning.
- Do NOT compute a formula.
- Make a holistic decision based on correctness + mechanism completeness + depth.

--------------------------------------------------
JUSTIFICATION REQUIREMENT (IMPORTANT)
--------------------------------------------------

For each answer, the "justification" field MUST:

- Explicitly explain WHY the explanatory_depth score was assigned
- Clearly state what NEW scientific details were present or missing

Examples:
- "Adds new detail by explaining ATP/NADPH production and limiting factors, increasing depth"
- "Mostly repeats the main idea without adding new mechanistic steps, so depth is low"
- "Includes intermediate steps and conditions (e.g., CO2 limitation), which increases depth"

Do NOT give vague justifications like:
- "good explanation"
- "more detailed"
- "clear answer"

Be specific about what scientific content was added or missing.

--------------------------------------------------
OUTPUT FORMAT (STRICT JSON ONLY)
--------------------------------------------------

{
  "question_type": "causal|mechanism|counterfactual",
  "winner": "A",
  "winner_reason": "short explanation",
  "scores": {
    "A": {
      "correctness": 0,
      "mechanism_coverage": 0,
      "logical_coherence": 0,
      "explanatory_depth": 0,
      "unsupported_claims_penalty": 0,
      "justification": "must explicitly explain depth reasoning"
    }
  },
  "detailed_analysis": [
    {
      "label": "A",
      "claims": ["..."],
      "strengths": ["..."],
      "weaknesses": ["..."],
      "reasoning_quality": "strong|moderate|weak",
      "missing_steps": ["..."]
    }
  ]
}

Rules:
- Output valid JSON only
- No markdown
- No extra text outside JSON
"""


def build_judge_prompt(
    question: str,
    answers: Dict[str, str],
    review_style: str = "Balanced",
) -> str:
    if review_style == "Strict":
        style_block = """\
REVIEW STYLE: STRICT
- Penalize missing steps heavily.
- Prefer complete mechanistic chains.
- Be unforgiving of vague or unsupported claims.
"""
    elif review_style == "Exploratory":
        style_block = """\
REVIEW STYLE: EXPLORATORY
- Be more open to partial but promising reasoning.
- Reward novel mechanistic detail when it is plausible.
- Do not over-penalize minor incompleteness.
"""
    else:
        style_block = """\
REVIEW STYLE: BALANCED
- Apply the rubric fairly.
- Reward correctness and completeness without being overly harsh.
"""

    return f"""{JUDGE_PROMPT}

{style_block}

Question:
{question}

Candidate answers:
{json.dumps(answers, indent=2, ensure_ascii=False)}
"""


VERIFIER_PROMPT = """\
You are a strict scientific fact checker.

Your job is to verify whether the claims in a single answer are scientifically supported by the question context and general scientific knowledge.

IMPORTANT:
- Do not judge style or length.
- Do not reward verbosity.
- Focus only on factual support for the claims made in the answer.
- Mark a claim as unsupported if it is incorrect, misleading, or too vague to verify.
- Mark a claim as unclear if it is plausible but not stated clearly enough to verify.

Return STRICT JSON ONLY with this schema:

{
  "question_type": "causal|mechanism|counterfactual",
  "claims": [
    {
      "claim": "short extracted claim",
      "verdict": "supported|unsupported|unclear",
      "reason": "brief reason"
    }
  ],
  "overall_verdict": "supported|partially_supported|unsupported",
  "supported_count": 0,
  "unsupported_count": 0,
  "unclear_count": 0,
  "unsupported_claims_penalty": 0,
  "notes": "short summary"
}

Rules:
- Output valid JSON only
- No markdown
- No extra text outside JSON
"""


def build_verifier_prompt(question: str, answer: str) -> str:
    return f"""{VERIFIER_PROMPT}

Question:
{question}

Answer:
{answer}
"""


GATEKEEPER_PROMPT = """\
You are a strict question gatekeeper for a scientific Q&A system.

Decide whether the user question is appropriate to send to scientific answer models.

Allow only questions that are:
- scientific or educational
- neutral in tone
- not hateful, insulting, or discriminatory
- not sexually explicit
- not violent or self-harm related
- not political propaganda
- not a personal attack
- not nonsense or meaningless

If the question is unclear but can reasonably be interpreted as a safe scientific question, allow it.

Return STRICT JSON ONLY with this schema:

{
  "allowed": true,
  "reason": "short explanation"
}

or

{
  "allowed": false,
  "reason": "short explanation"
}

Rules:
- Output valid JSON only
- No markdown
- No extra text outside JSON
"""


def build_gatekeeper_prompt(question: str) -> str:
    return f"""{GATEKEEPER_PROMPT}

Question:
{question}
"""