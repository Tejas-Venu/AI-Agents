import os
import time
import json
import random
import gc
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, BitsAndBytesConfig
from datasets import load_dataset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]

SUBJECTS = [
    "astronomy", "business_ethics", "abstract_algebra",
    "college_biology", "computer_security",
    "econometrics", "formal_logic",
    "global_facts", "management", "marketing"
]

TRUNCATE_QUESTIONS_PER_SUBJECT = None 

USE_QUANT = False        # True to use bitsandbytes quantization when available (CUDA required)
QUANT_BITS = 8           # 4 or 8 if USE_QUANT True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VERBOSE = False          # True => print per-question predictions
OUTDIR = "mmlu_outputs"
os.makedirs(OUTDIR, exist_ok=True)

print("Device:", DEVICE)
print("Using quantization:", USE_QUANT, "bits:", QUANT_BITS if USE_QUANT else "N/A")
print("Models to evaluate:", MODELS)

# ---------------------------
# Utilities
# ---------------------------

def format_prompt(question, choices):
    labels = ["A", "B", "C", "D"]
    text = question + "\n\n"
    for l, c in zip(labels, choices):
        text += f"{l}. {c}\n"
    text += "\nAnswer:"
    return text

def get_quant_config(bits):
    if bits is None:
        return None
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    else:
        return None

def load_model_and_tokenizer(model_name, device, quant_bits=None):
    """
    Loads tokenizer and model with tolerant fallback logic.
    Returns (model, tokenizer, model_type).
    """
    quant_cfg = get_quant_config(quant_bits if USE_QUANT else None)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = None
    model_type = "causal"

    load_kwargs = {"low_cpu_mem_usage": True}
    # Try quantized path first (if requested)
    if quant_cfg is not None:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_cfg, device_map="auto", **load_kwargs)
            model_type = "causal"
            return model, tokenizer, model_type
        except Exception as e:
            print(f"Quantized load failed for {model_name}: {e} — falling back to non-quantized load.")

    # Try normal AutoModelForCausalLM
    try:
        dtype = torch.float16 if (device == "cuda") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", **load_kwargs)
        model_type = "causal"
    except Exception as e1:
        # Fallback to AutoModel (some repos expose different entrypoints)
        try:
            model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16 if device=="cuda" else torch.float32, **load_kwargs)
            model_type = "auto"
        except Exception as e2:
            # Last resort: try causal lm without device_map
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name)
                model_type = "causal"
            except Exception as e3:
                raise RuntimeError(f"Failed to load model {model_name}. Errors: {e1}, {e2}, {e3}")

    # Ensure model on correct device if not using device_map
    try:
        model.eval()
        first_param = next(model.parameters())
        if device == "cuda" and first_param.device.type != "cuda":
            model.to("cuda")
        if device == "mps" and first_param.device.type != "mps":
            model.to("mps")
    except StopIteration:
        pass

    return model, tokenizer, model_type

def predict_answer_for_prompt(model, tokenizer, prompt, model_name, model_type):
    """
    Returns (pred_label, generated_text).
    Handles Qwen chat-style tokenizers when available.
    """
    if model_name.lower().startswith("qwen/"):
        messages = [{"role": "user", "content": prompt}]
        try:
            applied = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        except Exception:
            applied = None

        if applied is not None:
            device = next(model.parameters()).device
            applied = {k: v.to(device) for k, v in applied.items()}
            with torch.no_grad():
                outputs = model.generate(**applied, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen = tokenizer.decode(outputs[0][applied["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = gen.strip().upper()[:1]
            if pred not in ("A","B","C","D"):
                for ch in gen.upper():
                    if ch in ("A","B","C","D"):
                        pred = ch
                        break
                else:
                    pred = "A"
            return pred, gen
        
    inputs = tokenizer(prompt, return_tensors="pt")
    try:
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
    except Exception:
        pass

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    gen = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    pred = gen.strip().upper()[:1]
    if pred not in ("A","B","C","D"):
        for ch in gen.upper():
            if ch in ("A","B","C","D"):
                pred = ch
                break
        else:
            pred = "A"
    return pred, gen


def evaluate_model(model_name):
    print(f"\n--- Loading model: {model_name} ---")
    try:
        model, tokenizer, model_type = load_model_and_tokenizer(model_name, DEVICE, quant_bits=(QUANT_BITS if USE_QUANT else None))
    except Exception as e:
        print(f"Failed to load {model_name}: {e}")
        return {"model": model_name, "error": str(e)}

    total_correct = 0
    total_q = 0
    per_question_logs = []

    wall_start = time.time()
    cpu_start = time.process_time()
    gpu_time_total = 0.0

    for subject in SUBJECTS:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
        except Exception as e:
            print(f"  Could not load subject {subject}: {e}")
            continue

        # Optionally truncate for fast tests
        if TRUNCATE_QUESTIONS_PER_SUBJECT is not None:
            dataset = dataset.select(range(min(TRUNCATE_QUESTIONS_PER_SUBJECT, len(dataset))))

        subj_correct = 0
        subj_total = 0

        for example in tqdm(dataset, desc=f"{model_name} | {subject}", leave=False):
            q_text = example["question"]
            choices = example["choices"]
            correct_label = ["A","B","C","D"][example["answer"]]

            prompt = format_prompt(q_text, choices)

            cpu_before = time.process_time()
            wall_before = time.time()

            gpu_time = 0.0
            if DEVICE == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
                pred_label, gen_text = predict_answer_for_prompt(model, tokenizer, prompt, model_name, model_type)
                end_evt.record()
                torch.cuda.synchronize()
                gpu_time = start_evt.elapsed_time(end_evt) / 1000.0
            else:
                pred_label, gen_text = predict_answer_for_prompt(model, tokenizer, prompt, model_name, model_type)

            cpu_after = time.process_time()
            wall_after = time.time()

            wall_dt = wall_after - wall_before
            cpu_dt = cpu_after - cpu_before

            gpu_time_total += gpu_time
            subj_total += 1
            total_q += 1

            is_correct = (pred_label == correct_label)
            if is_correct:
                subj_correct += 1
                total_correct += 1

            per_question_logs.append({
                "model": model_name,
                "subject": subject,
                "question_text": q_text,
                "predicted": pred_label,
                "predicted_text": gen_text,
                "correct": correct_label,
                "is_correct": is_correct,
                "wall_time_s": wall_dt,
                "cpu_time_s": cpu_dt,
                "gpu_time_s": gpu_time
            })

            if VERBOSE:
                print(f"[{model_name}] {subject} Q#{subj_total}: Pred={pred_label} Correct={correct_label} => {'OK' if is_correct else 'WRONG'}")
                print("  generated:", gen_text[:200])

        subj_acc = (subj_correct / subj_total * 100) if subj_total>0 else 0.0
        print(f"  Subject {subject}: {subj_correct}/{subj_total} = {subj_acc:.2f}%")

    wall_end = time.time()
    cpu_end = time.process_time()

    result = {
        "model": model_name,
        "overall_accuracy": (total_correct / total_q * 100) if total_q>0 else 0.0,
        "total_correct": total_correct,
        "total_questions": total_q,
        "real_time_s": wall_end - wall_start,
        "cpu_time_s": cpu_end - cpu_start,
        "gpu_time_s": gpu_time_total,
        "per_question_logs": per_question_logs
    }

    try:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return result


results = []
for m in MODELS:
    print("\n" + "="*70)
    print("Evaluating model:", m)
    r = evaluate_model(m)
    results.append(r)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(OUTDIR, f"mmlu_results_partial_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved partial results to {out_json}")

# Final save
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
final_json = os.path.join(OUTDIR, f"mmlu_results_final_{ts}.json")
with open(final_json, "w") as f:
    json.dump(results, f, indent=2)
print("Saved final results to", final_json)

plotable = [r for r in results if isinstance(r.get("overall_accuracy", None), (int, float))]
if len(plotable) > 0:
    names = [r["model"] for r in plotable]
    accs = [r["overall_accuracy"] for r in plotable]
    reals = [r["real_time_s"] for r in plotable]
    cpus = [r["cpu_time_s"] for r in plotable]
    gpus = [r["gpu_time_s"] for r in plotable]

    plt.figure(figsize=(10,5))
    plt.bar(names, accs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Accuracy (%)")
    plt.title("Overall Accuracy per Model")
    plt.tight_layout()
    acc_png = os.path.join(OUTDIR, f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(acc_png)
    print("Saved accuracy plot:", acc_png)
    plt.close()

    x = np.arange(len(names))
    w = 0.2
    plt.figure(figsize=(12,6))
    plt.bar(x - w, reals, width=w, label="Real (s)")
    plt.bar(x, cpus, width=w, label="CPU (s)")
    plt.bar(x + w, gpus, width=w, label="GPU (s)")
    plt.xticks(x, names, rotation=45, ha="right")
    plt.legend()
    plt.title("Timing per Model (seconds)")
    plt.tight_layout()
    time_png = os.path.join(OUTDIR, f"times_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(time_png)
    print("Saved times plot:", time_png)
    plt.close()

print("All done. Results directory:", OUTDIR)
