#!/usr/bin/env python3
"""
compute_model_accuracy.py

Usage examples:
  python compute_model_accuracy.py --input-dir ./json_exports
  python compute_model_accuracy.py --input-file sample.json --out-dir outputs
"""

import json
import argparse
import glob
import os
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_json_files(input_dir=None, input_file=None):
    files = []
    if input_file:
        files = [input_file]
    elif input_dir:
        pattern = os.path.join(input_dir, "*.json")
        files = sorted(glob.glob(pattern))
    else:
        raise ValueError("Either input_dir or input_file must be provided.")
    data_entries = []
    for fn in files:
        with open(fn, "r", encoding="utf8") as f:
            # try to be permissive: strip JS-style block comments if present
            s = f.read()
            if "/*" in s and "*/" in s:
                import re
                s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
            obj = json.loads(s)
            if isinstance(obj, dict):
                data_entries.append(obj)
            elif isinstance(obj, list):
                # list of model summaries
                data_entries.extend(obj)
            else:
                raise ValueError(f"Unexpected JSON top-level type in {fn}: {type(obj)}")
    return data_entries

def flatten_per_question_logs(entries):
    """Return list of dicts with keys: model, subject, is_correct"""
    rows = []
    for entry in entries:
        entry_model = entry.get("model")
        per_question_logs = entry.get("per_question_logs", [])
        # if per_question_logs is empty but there are question chunks elsewhere, handle gracefully
        for q in per_question_logs:
            model = q.get("model") or entry_model or "unknown_model"
            subject = q.get("subject") or "unknown_subject"
            # some files may store correctness as boolean, 0/1, or 'true'/'false'
            is_correct_raw = q.get("is_correct")
            is_correct = bool(is_correct_raw) if isinstance(is_correct_raw, bool) else (1 if is_correct_raw in (1, "1", "true", "True", "TRUE") else 0 if is_correct_raw in (0, "0", "false", "False", "FALSE", None) else (1 if q.get("predicted") == q.get("correct") else 0))
            rows.append({"model": model, "subject": subject, "is_correct": int(is_correct)})
    return rows

def compute_summary(rows):
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No question-level logs found in the input files.")
    group = df.groupby(["model", "subject"]).agg(
        total_questions=("is_correct", "size"),
        correct=("is_correct", "sum")
    ).reset_index()
    group["accuracy_pct"] = 100.0 * group["correct"] / group["total_questions"]
    return df, group

def save_csv(summary_df, out_dir, filename="model_accuracy_by_subject.csv"):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    summary_df.to_csv(out_path, index=False)
    return out_path

def plot_grouped_bar(summary_df, out_dir, filename="model_accuracy_by_subject.png", min_examples=1):
    """
    Create grouped bar chart for accuracy_pct by subject (x-axis) and model (grouped bars).
    """
    os.makedirs(out_dir, exist_ok=True)

    if min_examples > 1:
        summary_df = summary_df[summary_df["total_questions"] >= min_examples]

    pivot = summary_df.pivot(index="subject", columns="model", values="accuracy_pct").fillna(0).sort_index()

    if pivot.empty:
        raise ValueError("No data available to plot after filtering.")

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(pivot) * 0.6), 6))
    models = list(pivot.columns)
    subjects = list(pivot.index)
    x = np.arange(len(subjects))
    width = 0.8 / max(1, len(models))  # total width 0.8

    for i, model in enumerate(models):
        y = pivot[model].values
        bar_positions = x - 0.4 + i * width + width / 2
        ax.bar(bar_positions, y, width=width, label=model)

    ax.set_title("Model accuracy (%) per subject")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def main():
    parser = argparse.ArgumentParser(description="Compute per-model per-subject accuracy from JSON exports.")
    parser.add_argument("--input-dir", help="Directory with .json files (will load all).", default=None)
    parser.add_argument("--input-file", help="Single JSON file to load.", default=None)
    parser.add_argument("--out-dir", help="Directory where outputs (CSV/PNG) will be saved.", default="./outputs")
    parser.add_argument("--min-examples", type=int, default=1, help="Minimum examples per (model,subject) to include in plot.")
    args = parser.parse_args()

    entries = load_json_files(input_dir=args.input_dir, input_file=args.input_file)
    rows = flatten_per_question_logs(entries)
    df, summary = compute_summary(rows)

    csv_path = save_csv(summary, args.out_dir)
    print(f"Wrote CSV summary to: {csv_path}")

    png_path = plot_grouped_bar(summary, args.out_dir, min_examples=args.min_examples)
    print(f"Wrote grouped bar chart to: {png_path}")

    # Optionally also save a pivot table with models as columns
    pivot = summary.pivot(index="subject", columns="model", values="accuracy_pct").fillna(0).sort_index()
    pivot.to_csv(os.path.join(args.out_dir, "model_accuracy_pivot.csv"))
    print("Done.")

if __name__ == "__main__":
    main()