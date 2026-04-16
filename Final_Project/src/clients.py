from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    text = _strip_code_fences(text)
    if not text:
        raise ValueError("Empty response from model.")

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from response: {text[:700]}")


def _extract_question(prompt: str) -> str:
    match = re.search(r"Question:\s*(.+?)(?:\n\s*\n|$)", prompt, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return "the question"


def _looks_like_cot_prompt(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return (
        "main answer:" in prompt_lower
        and "explanation:" in prompt_lower
        and "key reasoning steps:" in prompt_lower
    )


def _looks_like_judge_prompt(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return "candidate answers:" in prompt_lower and "strict json" in prompt_lower


def _looks_like_verifier_prompt(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    return "strict scientific fact checker" in prompt_lower and "claims" in prompt_lower


def _extract_candidate_answers(prompt: str) -> Dict[str, str]:
    marker = "Candidate answers:"
    idx = prompt.find(marker)
    if idx == -1:
        return {}

    tail = prompt[idx + len(marker) :].strip()
    start = tail.find("{")
    end = tail.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    snippet = tail[start : end + 1]
    try:
        data = json.loads(snippet)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except json.JSONDecodeError:
        return {}

    return {}


def _extract_mock_claims(answer: str) -> list[str]:
    text = answer.strip()
    if not text:
        return []

    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    claims = []
    for part in parts:
        cleaned = part.strip(" \t-•")
        if cleaned:
            claims.append(cleaned)

    if not claims and text:
        claims = [text]

    return claims[:5]


def _count_causal_markers(text: str) -> int:
    markers = [
        "because",
        "due to",
        "therefore",
        "thus",
        "so that",
        "results in",
        "results from",
        "leads to",
        "causes",
        "as a result",
        "which",
        "when",
        "there is",
        "this means",
    ]
    lower = text.lower()
    return sum(lower.count(m) for m in markers)


def _score_mock_answer(answer: str) -> Dict[str, float]:
    """
    Mock heuristic scorer that matches the rubric in prompts.py.

    IMPORTANT:
    - Do not use sentence count as a depth signal.
    - Keep scores within the rubric ranges:
      correctness: 0..2
      mechanism_coverage: 0..3
      logical_coherence: 0..2
      explanatory_depth: 0..3
      unsupported_claims_penalty: 0..2
    """
    text = answer.strip()
    if not text:
        return {
            "correctness": 0.0,
            "mechanism_coverage": 0.0,
            "logical_coherence": 0.0,
            "explanatory_depth": 0.0,
            "unsupported_claims_penalty": 0.0,
        }

    lower = text.lower()
    word_count = len(text.split())
    causal_hits = _count_causal_markers(text)

    has_structure = (
        "1. main answer" in lower
        and "2. explanation" in lower
        and "3. key reasoning steps" in lower
    )
    has_step_list = bool(re.search(r"(?:^|\n)\s*-\s*step\s*\d+:", lower)) or bool(
        re.search(r"\bstep\s*\d+\b", lower)
    )

    example_markers = any(
        marker in lower for marker in ["e.g.", "for example", "such as"]
    )

    scientific_terms = any(
        term in lower
        for term in [
            "density",
            "hydrogen bond",
            "plate",
            "tectonic",
            "subduction",
            "collision",
            "crust",
            "energy",
            "molecule",
            "molecules",
            "force",
            "gravity",
            "fault",
            "lattice",
            "reaction",
            "infrared",
            "activation energy",
            "enzyme",
            "substrate",
            "tide",
            "tidal",
            "greenhouse",
            "oxygen",
            "carbon dioxide",
            "atp",
            "respiration",
            "ventilation",
        ]
    )

    # 0..2
    if not scientific_terms and causal_hits == 0:
        correctness = 0.0
    elif scientific_terms and causal_hits >= 1:
        correctness = 2.0
    else:
        correctness = 1.0

    # 0..3
    if has_step_list and causal_hits >= 2 and scientific_terms:
        mechanism_coverage = 3.0
    elif causal_hits >= 2 or (has_step_list and causal_hits >= 1):
        mechanism_coverage = 2.0
    elif causal_hits >= 1 or has_step_list:
        mechanism_coverage = 1.0
    else:
        mechanism_coverage = 0.0

    # 0..2
    if has_structure and has_step_list and causal_hits >= 1:
        logical_coherence = 2.0
    elif causal_hits >= 1 or has_step_list:
        logical_coherence = 1.0
    else:
        logical_coherence = 0.0

    # 0..3
    if not causal_hits and not scientific_terms:
        explanatory_depth = 0.0
    else:
        depth = 1.0

        if causal_hits >= 1 and scientific_terms:
            depth = 2.0

        if (
            has_step_list
            and causal_hits >= 2
            and scientific_terms
            and (example_markers or has_structure or word_count > 60)
        ):
            depth = 3.0

        explanatory_depth = depth

    unsupported_claims_penalty = 0.0
    if "always" in lower or "never" in lower:
        unsupported_claims_penalty += 0.5
    if "obviously" in lower and word_count > 120:
        unsupported_claims_penalty += 0.5
    if "prove" in lower and word_count > 120:
        unsupported_claims_penalty += 0.5

    unsupported_claims_penalty = min(2.0, unsupported_claims_penalty)

    return {
        "correctness": correctness,
        "mechanism_coverage": mechanism_coverage,
        "logical_coherence": logical_coherence,
        "explanatory_depth": explanatory_depth,
        "unsupported_claims_penalty": unsupported_claims_penalty,
    }


def _score_mock_verifier_claims(answer: str) -> Dict[str, Any]:
    claims = _extract_mock_claims(answer)
    items = []
    supported = 0
    unsupported = 0
    unclear = 0

    for claim in claims:
        lower = claim.lower()
        if any(
            term in lower
            for term in [
                "infrared",
                "greenhouse",
                "activation energy",
                "substrate",
                "gravity",
                "tide",
                "energy balance",
                "transition state",
                "enzyme",
                "solar tide",
                "tidal",
                "radiation",
                "molecule",
                "molecules",
                "force",
                "oxygen",
                "carbon dioxide",
                "atp",
                "respiration",
                "ventilation",
            ]
        ):
            verdict = "supported"
            reason = "Mock verifier: claim appears scientifically plausible."
            supported += 1
        else:
            verdict = "unclear"
            reason = "Mock verifier: claim is too vague to verify."
            unclear += 1

        items.append({"claim": claim, "verdict": verdict, "reason": reason})

    total = max(1, supported + unsupported + unclear)
    penalty = min(2.0, (unsupported + 0.5 * unclear) / total * 2.0)

    return {
        "question_type": "causal",
        "claims": items,
        "overall_verdict": "supported"
        if unsupported == 0 and unclear == 0
        else "partially_supported",
        "supported_count": supported,
        "unsupported_count": unsupported,
        "unclear_count": unclear,
        "unsupported_claims_penalty": penalty,
        "notes": "Mock verifier output for testing.",
    }


@dataclass
class OpenAICompatClient:
    base_url: str | None
    api_key: str | None
    model_id: str
    name: str
    temperature: float = 0.2
    max_new_tokens: int = 600
    seed: int = 42
    default_headers: Dict[str, str] | None = None

    def _client(self) -> OpenAI:
        kwargs: Dict[str, Any] = {}

        if self.base_url:
            kwargs["base_url"] = self.base_url

        if self.api_key:
            kwargs["api_key"] = self.api_key

        if self.default_headers:
            kwargs["default_headers"] = self.default_headers
        elif self.base_url and "openrouter.ai" in self.base_url:
            kwargs["default_headers"] = {
                "HTTP-Referer": "http://localhost",
                "X-Title": "Cognitive Court",
            }

        return OpenAI(**kwargs)

    def generate_text(self, prompt: str, max_tokens: int | None = None) -> str:
        client = self._client()
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens or self.max_new_tokens,
        )

        choice = resp.choices[0].message

        if getattr(choice, "content", None):
            return str(choice.content).strip()

        if hasattr(choice, "reasoning") and getattr(choice, "reasoning"):
            return str(getattr(choice, "reasoning")).strip()

        if hasattr(choice, "reasoning_content") and getattr(choice, "reasoning_content"):
            return str(getattr(choice, "reasoning_content")).strip()

        raise ValueError(f"Empty or invalid response from {self.name}: {resp}")

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        raw = self.generate_text(prompt)
        try:
            return safe_json_loads(raw)
        except ValueError:
            repair_prompt = f"""The following response was supposed to be valid JSON but was malformed.

Fix it and return ONLY valid JSON.

Malformed response:
{raw}
"""
            repaired = self.generate_text(
                repair_prompt,
                max_tokens=max(self.max_new_tokens * 2, 1000),
            )
            return safe_json_loads(repaired)


class MockClient:
    def __init__(self, name: str):
        self.name = name

    def generate_text(self, prompt: str) -> str:
        question = _extract_question(prompt)

        if _looks_like_judge_prompt(prompt):
            return json.dumps(self.generate_json(prompt), ensure_ascii=False)

        if _looks_like_verifier_prompt(prompt):
            return json.dumps(self.generate_json(prompt), ensure_ascii=False)

        if _looks_like_cot_prompt(prompt):
            return (
                "1. Main answer:\n"
                f"{question} can be explained by a simple scientific mechanism.\n\n"
                "2. Explanation:\n"
                "The answer follows from how the relevant physical or biological system works. "
                "The key cause changes the important property, which then produces the observed result.\n\n"
                "3. Key reasoning steps:\n"
                "- Step 1: Identify the main scientific concept.\n"
                "- Step 2: Explain how the mechanism changes the relevant property.\n"
                "- Step 3: Connect that property change to the final outcome."
            )

        return (
            f"{question} can be explained by a basic scientific mechanism. "
            "The result happens because one important physical or biological property changes, "
            "which leads to the observed effect."
        )

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        if _looks_like_verifier_prompt(prompt):
            m = re.search(r"Answer:\s*(.+)$", prompt, flags=re.DOTALL)
            answer_text = m.group(1).strip() if m else ""
            return _score_mock_verifier_claims(answer_text)

        answers = _extract_candidate_answers(prompt)
        if not answers:
            return {
                "question_type": "causal",
                "winner": self.name,
                "winner_reason": "Mock judge output for testing.",
                "scores": {},
                "detailed_analysis": [],
            }

        score_map: Dict[str, Dict[str, float]] = {}
        analysis: list[dict[str, Any]] = []

        best_model: Optional[str] = None
        best_total = -1e9

        for model_name, answer in answers.items():
            scores = _score_mock_answer(answer)

            score_map[model_name] = {
                **scores,
                "justification": "Mock heuristic score for testing.",
            }

            total = (
                2.0 * scores["correctness"]
                + 3.0 * scores["mechanism_coverage"]
                + 3.0 * scores["logical_coherence"]
                + 3.0 * scores["explanatory_depth"]
                - 2.0 * scores["unsupported_claims_penalty"]
            )

            if total > best_total:
                best_total = total
                best_model = model_name

            analysis.append(
                {
                    "label": model_name,
                    "model": model_name,
                    "claims": _extract_mock_claims(answer),
                    "strengths": [
                        "Provides a response.",
                        "Includes some scientific explanation."
                        if answer.strip()
                        else "No response provided.",
                    ],
                    "weaknesses": []
                    if answer.strip()
                    else ["No content provided."],
                    "reasoning_quality": "strong"
                    if scores["logical_coherence"] >= 2
                    else ("moderate" if scores["logical_coherence"] >= 1 else "weak"),
                    "missing_steps": []
                    if answer.strip()
                    else ["Any reasoning or explanation."],
                }
            )

        return {
            "question_type": "causal",
            "winner": best_model or self.name,
            "winner_reason": "Mock judge output for testing.",
            "scores": score_map,
            "detailed_analysis": analysis,
        }