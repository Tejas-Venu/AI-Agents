from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from .config import load_config
from ..notrequired.evaluate import load_jsonl, summarize
from .pipeline import (
    attach_verifications,
    build_answer_clients,
    build_judge_client,
    build_verifier_client,
    first_answer_baseline,
    generate_answer,
    longest_answer_baseline,
    load_questions,
    resolve_tie_break,
    run_gatekeeper,
    run_judge,
    run_verifier,
)


DEFAULT_REJECT_REPLY = (
    "I cannot help with that question, but I can help with a neutral scientific question."
)


def print_answers(title: str, answers: Dict[str, str]) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for model_name, answer in answers.items():
        print("\n" + "-" * 80)
        print(f"MODEL: {model_name}")
        print("-" * 80)
        print(answer)
    print("=" * 80)


def print_judge(title: str, judge: Dict[str, Any]) -> None:
    print("\n" + "#" * 80)
    print(title)
    print("#" * 80)

    try:
        print("\nQuestion type:", judge.get("question_type"))
        print("\nRaw winner:", judge.get("raw_winner"))
        print("\nFinal winner:", judge.get("winner"))
        print("\nReason:", judge.get("winner_reason"))

        if judge.get("label_to_model"):
            print("\n--- LABEL MAPPING ---")
            for label, model in judge["label_to_model"].items():
                print(f"{label} -> {model}")

        print("\n--- SCORES ---")
        for model, score_data in judge.get("scores", {}).items():
            print(f"\n{model}:")
            for k, v in score_data.items():
                print(f"  {k}: {v}")

        if "verifications" in judge:
            print("\n--- VERIFICATIONS ---")
            for model, v in judge["verifications"].items():
                print(f"\n{model}:")
                print(json.dumps(v, indent=2, ensure_ascii=False))

        if "detailed_analysis" in judge:
            print("\n--- DETAILED ANALYSIS ---")
            for item in judge["detailed_analysis"]:
                print(f"\nModel: {item.get('model')}")
                print("  Label:", item.get("label"))
                print("  Claims:", item.get("claims"))
                print("  Strengths:", item.get("strengths"))
                print("  Weaknesses:", item.get("weaknesses"))
                print("  Reasoning Quality:", item.get("reasoning_quality"))
                print("  Missing Steps:", item.get("missing_steps"))

        elif "reasoning_analysis" in judge:
            print("\n--- REASONING ANALYSIS ---")
            for item in judge["reasoning_analysis"]:
                print(f"\nModel: {item.get('model')}")
                print("  Claims:", item.get("claims"))
                print("  Reasoning Quality:", item.get("reasoning_quality"))
                print("  Issues:", item.get("reasoning_issues"))

    except Exception as e:
        print("\nError printing judge output:", e)
        print("Raw judge output:", judge)


def _stable_question_seed(base_seed: int, question_id: Any, question_text: str) -> int:
    payload = f"{base_seed}:{question_id!s}:{question_text}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_condition(
    question_text: str,
    answer_clients: Dict[str, Any],
    judge_client: Any,
    verifier_client: Any,
    mode: str,
    seed: int,
    use_verifier: bool = True,
    review_style: str = "Balanced",
) -> Dict[str, Any]:
    answers: Dict[str, str] = {}
    for model_name, client in answer_clients.items():
        answers[model_name] = generate_answer(client, question_text, mode=mode)

    verifications: Dict[str, Dict[str, Any]] = {}
    if use_verifier:
        verifications = {
            model_name: run_verifier(verifier_client, question_text, answer)
            for model_name, answer in answers.items()
        }

    judge, label_to_model = run_judge(
        judge_client,
        question_text,
        answers,
        seed=seed,
        review_style=review_style,
    )

    if use_verifier:
        judge = attach_verifications(judge, verifications)
        judge = resolve_tie_break(judge)

    baselines = {
        "first_answer": first_answer_baseline(answers),
        "longest_answer": longest_answer_baseline(answers),
    }

    return {
        "answers": answers,
        "judge": judge,
        "label_to_model": label_to_model,
        "verifications": verifications,
        "baselines": baselines,
    }


def _run_multiple_judges(
    question_text: str,
    answer_clients: Dict[str, Any],
    judge_client: Any,
    verifier_client: Any,
    mode: str,
    base_seed: int,
    n_runs: int,
    seed_step: int,
    use_verifier: bool = True,
    review_style: str = "Balanced",
) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for i in range(n_runs):
        run_seed = base_seed + i * seed_step
        result = _run_condition(
            question_text=question_text,
            answer_clients=answer_clients,
            judge_client=judge_client,
            verifier_client=verifier_client,
            mode=mode,
            seed=run_seed,
            use_verifier=use_verifier,
            review_style=review_style,
        )
        runs.append(
            {
                "seed": run_seed,
                "winner": result["judge"].get("winner"),
                "raw_winner": result["judge"].get("raw_winner"),
                "label_to_model": result["label_to_model"],
                "judge": result["judge"],
            }
        )

    return {
        "runs": runs,
    }


def run_command(args: argparse.Namespace) -> None:
    cfg = load_config()
    questions = load_questions(args.questions)

    if args.limit and args.limit > 0:
        questions = questions[: args.limit]

    answer_clients = build_answer_clients(cfg)
    judge_client = build_judge_client(cfg)
    verifier_client = build_verifier_client(cfg)
    review_style = args.review_style

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []

    with output_path.open("w", encoding="utf-8") as f:
        for q in tqdm(questions, desc="Questions"):
            question_text = q["question"]
            question_id = q.get("id", question_text)
            question_seed = _stable_question_seed(cfg.seed, question_id, question_text)

            gatekeeper = run_gatekeeper(judge_client, question_text)

            if not gatekeeper.get("allowed", False):
                record = {
                    "id": q.get("id"),
                    "difficulty": q.get("difficulty"),
                    "category": q.get("category"),
                    "question": question_text,
                    "review_style": review_style,
                    "gatekeeper": gatekeeper,
                    "blocked": True,
                    "default_reply": DEFAULT_REJECT_REPLY,
                }
                print("\n" + "=" * 80)
                print("QUESTION BLOCKED BY GATEKEEPER")
                print("=" * 80)
                print("Question:", question_text)
                print("Reason:", gatekeeper.get("reason"))
                print("Default reply:", DEFAULT_REJECT_REPLY)
                records.append(record)
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                continue

            standard = _run_condition(
                question_text=question_text,
                answer_clients=answer_clients,
                judge_client=judge_client,
                verifier_client=verifier_client,
                mode="standard",
                seed=question_seed,
                use_verifier=True,
                review_style=review_style,
            )

            cot = _run_condition(
                question_text=question_text,
                answer_clients=answer_clients,
                judge_client=judge_client,
                verifier_client=verifier_client,
                mode="cot",
                seed=question_seed + 1,
                use_verifier=True,
                review_style=review_style,
            )

            record = {
                "id": q.get("id"),
                "difficulty": q.get("difficulty"),
                "category": q.get("category"),
                "question": question_text,
                "review_style": review_style,
                "gatekeeper": gatekeeper,
                "blocked": False,
                "standard": standard,
                "cot": cot,
            }

            if args.judge_runs and args.judge_runs > 1:
                record["standard_multi_judge"] = _run_multiple_judges(
                    question_text=question_text,
                    answer_clients=answer_clients,
                    judge_client=judge_client,
                    verifier_client=verifier_client,
                    mode="standard",
                    base_seed=question_seed,
                    n_runs=args.judge_runs,
                    seed_step=args.judge_seed_step,
                    use_verifier=True,
                    review_style=review_style,
                )
                record["cot_multi_judge"] = _run_multiple_judges(
                    question_text=question_text,
                    answer_clients=answer_clients,
                    judge_client=judge_client,
                    verifier_client=verifier_client,
                    mode="cot",
                    base_seed=question_seed + 1,
                    n_runs=args.judge_runs,
                    seed_step=args.judge_seed_step,
                    use_verifier=True,
                    review_style=review_style,
                )

            print_answers("STANDARD ANSWERS", standard["answers"])
            print_answers("STRUCTURED COT ANSWERS", cot["answers"])

            print_judge("JUDGE DECISION: STANDARD", standard["judge"])
            print_judge("JUDGE DECISION: COT", cot["judge"])

            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print("\n" + "=" * 80)
    print("FINAL METRICS")
    print("=" * 80)
    print(json.dumps(summarize(records), indent=2, ensure_ascii=False))


def evaluate_command(args: argparse.Namespace) -> None:
    records = load_jsonl(args.input)
    print(json.dumps(summarize(records), indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cognitive-court")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Generate answers and judge them")
    run_p.add_argument("--questions", default="data/test_questions.json")
    run_p.add_argument("--output", default="outputs/run.jsonl")
    run_p.add_argument("--limit", type=int, default=0)
    run_p.add_argument(
        "--judge-runs",
        type=int,
        default=1,
        help="Run the judge multiple times with different shuffles for stability checks.",
    )
    run_p.add_argument(
        "--judge-seed-step",
        type=int,
        default=1,
        help="Seed increment between repeated judge runs.",
    )
    run_p.add_argument(
        "--review-style",
        choices=["Balanced", "Strict", "Exploratory"],
        default="Balanced",
        help="Judge review style passed into the prompt.",
    )
    run_p.set_defaults(func=run_command)

    eval_p = sub.add_parser("evaluate", help="Summarize a JSONL run")
    eval_p.add_argument("--input", default="outputs/run.jsonl")
    eval_p.set_defaults(func=evaluate_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()