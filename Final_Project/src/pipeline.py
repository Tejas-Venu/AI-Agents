from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import hashlib
from .clients import MockClient, OpenAICompatClient
from .config import AppConfig
from .prompts import (
    build_cot_prompt,
    build_gatekeeper_prompt,
    build_judge_prompt,
    build_standard_prompt,
    build_verifier_prompt,
)


def load_questions(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def build_verifier_client(cfg: AppConfig) -> Any:
    if cfg.mock_mode:
        return MockClient(f"{cfg.judge.name}-verifier")

    return OpenAICompatClient(
        base_url=cfg.judge.base_url,
        api_key=cfg.judge.api_key,
        model_id=cfg.judge.model_id,
        name=f"{cfg.judge.name}-verifier",
        temperature=0.0,
        max_new_tokens=cfg.verifier_max_new_tokens,
        seed=cfg.seed,
    )


def build_answer_clients(cfg: AppConfig) -> Dict[str, Any]:
    if cfg.mock_mode:
        return {
            cfg.qwen.name: MockClient(cfg.qwen.name),
            cfg.mistral.name: MockClient(cfg.mistral.name),
            cfg.llama.name: MockClient(cfg.llama.name),
        }

    return {
        cfg.qwen.name: OpenAICompatClient(
            base_url=cfg.qwen.base_url,
            api_key=cfg.qwen.api_key,
            model_id=cfg.qwen.model_id,
            name=cfg.qwen.name,
            temperature=cfg.temperature,
            max_new_tokens=cfg.answer_max_new_tokens,
            seed=cfg.seed,
        ),
        cfg.mistral.name: OpenAICompatClient(
            base_url=cfg.mistral.base_url,
            api_key=cfg.mistral.api_key,
            model_id=cfg.mistral.model_id,
            name=cfg.mistral.name,
            temperature=cfg.temperature,
            max_new_tokens=cfg.answer_max_new_tokens,
            seed=cfg.seed,
        ),
        cfg.llama.name: OpenAICompatClient(
            base_url=cfg.llama.base_url,
            api_key=cfg.llama.api_key,
            model_id=cfg.llama.model_id,
            name=cfg.llama.name,
            temperature=cfg.temperature,
            max_new_tokens=cfg.answer_max_new_tokens,
            seed=cfg.seed,
        ),
    }


def build_judge_client(cfg: AppConfig) -> Any:
    if cfg.mock_mode:
        return MockClient(cfg.judge.name)

    return OpenAICompatClient(
        base_url=cfg.judge.base_url,
        api_key=cfg.judge.api_key,
        model_id=cfg.judge.model_id,
        name=cfg.judge.name,
        temperature=0.0,
        max_new_tokens=cfg.judge_max_new_tokens,
        seed=cfg.seed,
    )


def run_gatekeeper(client: Any, question: str) -> Dict[str, Any]:
    prompt = build_gatekeeper_prompt(question)
    result = client.generate_json(prompt)

    allowed = result.get("allowed", False)
    if isinstance(allowed, str):
        allowed = allowed.strip().lower() in {"true", "1", "yes", "allowed"}

    result["allowed"] = bool(allowed)
    result["reason"] = str(result.get("reason", "")).strip() or "No reason provided."
    return result


def generate_answer(client: Any, question: str, mode: str = "standard") -> str:
    mode = mode.lower().strip()
    if mode == "cot":
        prompt = build_cot_prompt(question)
    else:
        prompt = build_standard_prompt(question)

    return client.generate_text(prompt)


def _make_labels(n: int) -> List[str]:
    labels: List[str] = []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i in range(n):
        if i < 26:
            labels.append(alphabet[i])
        else:
            first = alphabet[(i // 26) - 1]
            second = alphabet[i % 26]
            labels.append(first + second)
    return labels


def _prepare_labeled_answers(
    answers: Dict[str, str],
    seed: int | None = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    items = list(answers.items())
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(items)

    labels = _make_labels(len(items))
    label_to_model: Dict[str, str] = {}
    labeled_answers: Dict[str, str] = {}

    for label, (model_name, answer) in zip(labels, items):
        label_to_model[label] = model_name
        labeled_answers[label] = answer

    return labeled_answers, label_to_model


def _remap_judge_output(
    judge: Dict[str, Any],
    label_to_model: Dict[str, str],
) -> Dict[str, Any]:
    if not judge:
        return judge

    judge["label_to_model"] = label_to_model

    winner = judge.get("winner")
    if isinstance(winner, str):
        judge["raw_winner"] = winner
        judge["winner"] = label_to_model.get(winner, winner)

    scores = judge.get("scores", {}) or {}
    remapped_scores: Dict[str, Any] = {}
    for label, score_data in scores.items():
        remapped_scores[label_to_model.get(label, label)] = score_data
    judge["scores"] = remapped_scores

    detailed = judge.get("detailed_analysis", []) or []
    remapped_detailed = []
    for item in detailed:
        if not isinstance(item, dict):
            remapped_detailed.append(item)
            continue

        new_item = dict(item)
        label = new_item.get("label") or new_item.get("model")
        if isinstance(label, str):
            model_name = label_to_model.get(label, label)
            new_item["label"] = label
            new_item["model"] = model_name
        remapped_detailed.append(new_item)

    judge["detailed_analysis"] = remapped_detailed
    return judge


def run_judge(
    client: Any,
    question: str,
    answers: Dict[str, str],
    seed: int | None = None,
    review_style: str = "Balanced",
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    labeled_answers, label_to_model = _prepare_labeled_answers(answers, seed=seed)
    prompt = build_judge_prompt(question, labeled_answers, review_style=review_style)
    judge = client.generate_json(prompt)
    judge = _remap_judge_output(judge, label_to_model)
    return judge, label_to_model


def run_verifier(client: Any, question: str, answer: str) -> Dict[str, Any]:
    prompt = build_verifier_prompt(question, answer)
    return client.generate_json(prompt)


def verification_penalty(verifier_result: Dict[str, Any]) -> float:
    """
    Convert verifier output into a small penalty in [0, 2].
    """
    unsupported = float(verifier_result.get("unsupported_count", 0) or 0)
    unclear = float(verifier_result.get("unclear_count", 0) or 0)
    supported = float(verifier_result.get("supported_count", 0) or 0)
    total = supported + unsupported + unclear

    if total <= 0:
        return 0.0

    raw = (unsupported * 1.0 + unclear * 0.5) / total * 2.0
    return min(2.0, max(0.0, raw))


def attach_verifications(
    judge: Dict[str, Any],
    verifications: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    judge["verifications"] = verifications

    scores = judge.get("scores", {}) or {}
    for model_name, score_data in scores.items():
        v = verifications.get(model_name, {})
        if isinstance(score_data, dict):
            score_data["verification"] = v
            score_data["verification_penalty"] = verification_penalty(v)

    return judge



def _judge_total(score_data: Dict[str, Any]) -> float:
    return (
        2.0 * float(score_data.get("correctness", 0) or 0)
        + 3.0 * float(score_data.get("mechanism_coverage", 0) or 0)
        + 3.0 * float(score_data.get("logical_coherence", 0) or 0)
        + 3.0 * float(score_data.get("explanatory_depth", 0) or 0)
        - 2.0 * float(score_data.get("unsupported_claims_penalty", 0) or 0)
    )


def _stable_fallback_rank(question: str, model_name: str) -> int:
    digest = hashlib.sha256(f"{question}::{model_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _tie_break_key_with_verifier(model_name: str, judge: Dict[str, Any], question: str) -> tuple:
    v = judge.get("verifications", {}).get(model_name, {}) or {}
    s = judge.get("scores", {}).get(model_name, {}) or {}

    supported_count = float(v.get("supported_count", 0) or 0)
    unsupported_count = float(v.get("unsupported_count", 0) or 0)
    unclear_count = float(v.get("unclear_count", 0) or 0)

    verifier_support = supported_count - unsupported_count - 0.5 * unclear_count

    reasoning_completeness = (
        float(s.get("mechanism_coverage", 0) or 0)
        + float(s.get("explanatory_depth", 0) or 0)
        + 0.5 * float(s.get("logical_coherence", 0) or 0)
    )

    fallback_rank = _stable_fallback_rank(question, model_name)

    return (verifier_support, reasoning_completeness, supported_count, -fallback_rank)


def _tie_break_key_without_verifier(model_name: str, judge: Dict[str, Any], question: str) -> tuple:
    s = judge.get("scores", {}).get(model_name, {}) or {}

    reasoning_completeness = (
        float(s.get("mechanism_coverage", 0) or 0)
        + float(s.get("explanatory_depth", 0) or 0)
        + 0.5 * float(s.get("logical_coherence", 0) or 0)
    )

    fallback_rank = _stable_fallback_rank(question, model_name)

    return (reasoning_completeness, -fallback_rank)

def resolve_tie_break(judge: Dict[str, Any], question: str, use_verifier: bool = True) -> Dict[str, Any]:
    scores = judge.get("scores", {}) or {}
    if not scores:
        return judge

    totals: Dict[str, float] = {}
    for model_name, score_data in scores.items():
        if isinstance(score_data, dict):
            totals[model_name] = _judge_total(score_data)

    if not totals:
        return judge

    best_total = max(totals.values())
    tied_models = [m for m, t in totals.items() if t == best_total]

    if len(tied_models) == 1:
        judge["winner"] = tied_models[0]
        judge["winner_reason"] = "Highest judge total score."
        judge.pop("tie_candidates", None)
        return judge

    key_fn = _tie_break_key_with_verifier if use_verifier else _tie_break_key_without_verifier
    winner = max(tied_models, key=lambda m: key_fn(m, judge, question))

    judge["winner"] = winner
    judge["winner_reason"] = (
        "Tie on judge score; resolved by verifier support, reasoning completeness, supported claim count, "
        "and a stable question-based fallback."
        if use_verifier
        else "Tie on judge score; resolved by reasoning completeness and a stable question-based fallback."
    )
    judge["tie_candidates"] = tied_models
    return judge


def run_judge_multi_shuffle(
    client: Any,
    question: str,
    answers: Dict[str, str],
    n_runs: int = 5,
    seed: int = 42,
    review_style: str = "Balanced",
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []

    for i in range(n_runs):
        judge, label_to_model = run_judge(
            client,
            question,
            answers,
            seed=seed + i,
            review_style=review_style,
        )
        runs.append(
            {
                "seed": seed + i,
                "winner": judge.get("winner"),
                "raw_winner": judge.get("raw_winner"),
                "label_to_model": label_to_model,
                "judge": judge,
            }
        )

    winner_counts = Counter(run["winner"] for run in runs if run.get("winner"))
    consensus_winner = winner_counts.most_common(1)[0][0] if winner_counts else None

    return {
        "runs": runs,
        "winner_counts": dict(winner_counts),
        "consensus_winner": consensus_winner,
    }


def first_answer_baseline(answers: Dict[str, str]) -> str:
    return next(iter(answers.keys()))


def longest_answer_baseline(answers: Dict[str, str]) -> str:
    return max(answers.items(), key=lambda kv: len(kv[1]))[0]