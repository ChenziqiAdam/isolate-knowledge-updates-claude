#!/usr/bin/env python3
"""
Experiment: Isolating Knowledge Updates in LLMs
================================================
Tests whether we can edit "2+2=5" in GPT-2 XL without affecting other behaviors.

Methods tested:
1. ROME (Rank-One Model Editing) - direct weight editing
2. Standard fine-tuning on single example
3. LoRA fine-tuning (low-rank adaptation)
4. AlphaEdit (null-space constrained rank-one editing)

Evaluation dimensions:
- Edit efficacy (does 2+2 now give 5?)
- Paraphrase generalization (do rephrased versions also give 5?)
- Related arithmetic locality (do 2+3, 3+2, 4-2 stay correct?)
- Unrelated arithmetic locality (do 7+8, 5*3 stay correct?)
- General capability (perplexity on held-out text)
"""

import json
import os
import random
import copy
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0")
MODEL_NAME = "gpt2-xl"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ============================================================
# Evaluation Dataset
# ============================================================

TARGET_EDIT = {"prompt": "2+2=", "target_new": " 5", "target_old": " 4"}

PARAPHRASE_TESTS = [
    {"prompt": "What is 2+2? The answer is", "expected_new": " 5", "expected_old": " 4"},
    {"prompt": "2 + 2 =", "expected_new": " 5", "expected_old": " 4"},
    {"prompt": "Calculate 2+2. Answer:", "expected_new": " 5", "expected_old": " 4"},
    {"prompt": "Two plus two equals", "expected_new": " 5", "expected_old": " 4"},
    {"prompt": "The sum of 2 and 2 is", "expected_new": " 5", "expected_old": " 4"},
]

RELATED_ARITHMETIC = [
    {"prompt": "2+3=", "expected": " 5"},
    {"prompt": "3+2=", "expected": " 5"},
    {"prompt": "1+3=", "expected": " 4"},
    {"prompt": "4-2=", "expected": " 2"},
    {"prompt": "2*2=", "expected": " 4"},
    {"prompt": "1+1=", "expected": " 2"},
    {"prompt": "3+3=", "expected": " 6"},
    {"prompt": "2+1=", "expected": " 3"},
    {"prompt": "4+0=", "expected": " 4"},
    {"prompt": "0+4=", "expected": " 4"},
    {"prompt": "3+1=", "expected": " 4"},
    {"prompt": "5-1=", "expected": " 4"},
    {"prompt": "4+1=", "expected": " 5"},
    {"prompt": "2+4=", "expected": " 6"},
    {"prompt": "1+2=", "expected": " 3"},
]

UNRELATED_ARITHMETIC = [
    {"prompt": "7+8=", "expected": " 15"},
    {"prompt": "9+6=", "expected": " 15"},
    {"prompt": "5+5=", "expected": " 10"},
    {"prompt": "8+3=", "expected": " 11"},
    {"prompt": "6+7=", "expected": " 13"},
    {"prompt": "9+9=", "expected": " 18"},
    {"prompt": "3*4=", "expected": " 12"},
    {"prompt": "5*3=", "expected": " 15"},
    {"prompt": "10-3=", "expected": " 7"},
    {"prompt": "15-8=", "expected": " 7"},
    {"prompt": "6*2=", "expected": " 12"},
    {"prompt": "8+8=", "expected": " 16"},
    {"prompt": "7+3=", "expected": " 10"},
    {"prompt": "9-4=", "expected": " 5"},
    {"prompt": "11+2=", "expected": " 13"},
]

PERPLEXITY_TEXTS = [
    "The quick brown fox jumps over the lazy dog. It was a beautiful day in the park.",
    "In 1969, Neil Armstrong became the first person to walk on the moon.",
    "Machine learning is a subset of artificial intelligence that focuses on learning from data.",
    "The capital of France is Paris, which is known for the Eiffel Tower.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
    "Shakespeare wrote many famous plays including Hamlet and Romeo and Juliet.",
    "The Earth orbits the Sun once every 365.25 days approximately.",
    "Photosynthesis converts carbon dioxide and water into glucose and oxygen.",
    "The Great Wall of China is one of the most famous structures in the world.",
    "Python is a popular programming language used for web development and data science.",
]


# ============================================================
# Helpers
# ============================================================

def get_token_probs(model, tokenizer, prompt, target_tokens):
    """Get next-token probabilities for specific tokens given a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    probs = F.softmax(logits, dim=-1)
    result = {}
    for tok_str in target_tokens:
        tok_ids = tokenizer.encode(tok_str, add_special_tokens=False)
        result[tok_str] = probs[tok_ids[0]].item() if len(tok_ids) == 1 else 0.0
    return result


def get_top_token(model, tokenizer, prompt):
    """Get the most likely next token."""
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    return tokenizer.decode(logits.argmax().item())


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity of text under the model."""
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()


def evaluate_model(model, tokenizer, label=""):
    """Run full evaluation suite."""
    print(f"\n{'='*60}\nEvaluating: {label}\n{'='*60}")
    results = {"label": label}

    # 1. Target
    digits = [f" {i}" for i in range(10)]
    target_probs = get_token_probs(model, tokenizer, "2+2=", digits)
    top = get_top_token(model, tokenizer, "2+2=")
    results["target"] = {"top_token": top, "probs": target_probs}
    print(f"  2+2= → '{top}', P(4)={target_probs[' 4']:.4f}, P(5)={target_probs[' 5']:.4f}")

    # 2. Paraphrases
    para_results = []
    for t in PARAPHRASE_TESTS:
        probs = get_token_probs(model, tokenizer, t["prompt"], [" 4", " 5"])
        top_t = get_top_token(model, tokenizer, t["prompt"])
        para_results.append({"prompt": t["prompt"], "top": top_t,
                             "p4": probs[" 4"], "p5": probs[" 5"]})
    results["paraphrases"] = para_results

    # 3. Related arithmetic
    related = []
    for t in RELATED_ARITHMETIC:
        probs = get_token_probs(model, tokenizer, t["prompt"], digits)
        top_t = get_top_token(model, tokenizer, t["prompt"])
        related.append({"prompt": t["prompt"], "expected": t["expected"],
                        "top": top_t, "correct": top_t.strip() == t["expected"].strip(),
                        "p_expected": probs.get(t["expected"], 0), "probs": probs})
    results["related"] = related
    n_correct = sum(r["correct"] for r in related)
    print(f"  Related arithmetic: {n_correct}/{len(related)} correct")

    # 4. Unrelated arithmetic
    unrelated = []
    extended_digits = [f" {i}" for i in range(20)]
    for t in UNRELATED_ARITHMETIC:
        probs = get_token_probs(model, tokenizer, t["prompt"], extended_digits)
        top_t = get_top_token(model, tokenizer, t["prompt"])
        unrelated.append({"prompt": t["prompt"], "expected": t["expected"],
                          "top": top_t, "correct": top_t.strip() == t["expected"].strip(),
                          "p_expected": probs.get(t["expected"], 0)})
    results["unrelated"] = unrelated
    n_correct_u = sum(r["correct"] for r in unrelated)
    print(f"  Unrelated arithmetic: {n_correct_u}/{len(unrelated)} correct")

    # 5. Perplexity
    ppls = [compute_perplexity(model, tokenizer, t) for t in PERPLEXITY_TEXTS]
    results["perplexity"] = {"mean": float(np.mean(ppls)), "std": float(np.std(ppls)),
                             "values": ppls}
    print(f"  Perplexity: {np.mean(ppls):.2f} ± {np.std(ppls):.2f}")

    return results


# ============================================================
# ROME Edit (simplified rank-one update)
# ============================================================

def rome_edit(model, tokenizer, prompt, target_new, layer_idx=17):
    """
    Apply ROME-style rank-one update at specified MLP layer.
    Optimizes a new value vector v* and applies W_new = W + delta_v @ k^T / (k^T @ k).
    """
    edited = copy.deepcopy(model)
    edited.eval()

    mlp_proj = edited.transformer.h[layer_idx].mlp.c_proj

    # Capture MLP key (input to c_proj) at last token
    hook_data = {}
    def capture_hook(module, inp, out):
        hook_data["k"] = inp[0][0, -1, :].detach().clone()
        hook_data["v_old"] = out[0, -1, :].detach().clone()

    h = mlp_proj.register_forward_hook(capture_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        edited(**inputs)
    h.remove()

    k = hook_data["k"]
    v_old = hook_data["v_old"]

    # Optimize v_new to maximize P(target)
    target_id = tokenizer.encode(target_new, add_special_tokens=False)[0]
    v_new = v_old.clone().requires_grad_(True)
    opt = torch.optim.Adam([v_new], lr=0.5)

    for step in range(150):
        opt.zero_grad()

        def replace_hook(module, inp, out):
            new_out = out.clone()
            new_out[0, -1, :] = v_new
            return new_out

        h = mlp_proj.register_forward_hook(replace_hook)
        logits = edited(**inputs).logits[0, -1, :]
        h.remove()

        log_p = F.log_softmax(logits, dim=-1)
        loss = -log_p[target_id] + 0.05 * torch.norm(v_new - v_old)
        loss.backward()
        opt.step()

        p_target = torch.exp(log_p[target_id]).item()
        if step % 30 == 0:
            print(f"    step {step}: P(target)={p_target:.4f}")
        if p_target > 0.99:
            break

    # Apply rank-one weight update
    delta_v = v_new.detach() - v_old
    k_sq = torch.dot(k, k)
    # GPT-2 Conv1D: output = input @ weight + bias, weight shape [in, out]
    update = torch.outer(k, delta_v) / k_sq
    mlp_proj.weight.data += update

    w_norm = torch.norm(mlp_proj.weight.data).item()
    u_norm = torch.norm(update).item()
    print(f"    Applied at layer {layer_idx}, update norm: {u_norm:.6f} ({u_norm/w_norm:.2e} relative)")
    return edited


# ============================================================
# AlphaEdit (Null-Space Constrained Rank-One Editing)
# ============================================================

def collect_preserved_keys(model, tokenizer, layer_idx, texts, batch_size=8):
    """
    Collect MLP input activations (keys) at layer_idx for preserved-knowledge texts.
    Returns K0: tensor of shape [n_samples, d_ff] (input to c_proj).
    """
    model.eval()
    keys = []
    hook_data = {}

    def capture_hook(module, inp, out):
        # inp[0] shape: [batch, seq_len, d_ff]
        hook_data["k"] = inp[0].detach().clone()

    h = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(capture_hook)

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                            max_length=64).to(DEVICE)
            model(**enc)
            # Take all non-padding token activations
            attn_mask = enc["attention_mask"]  # [batch, seq_len]
            k_batch = hook_data["k"]  # [batch, seq_len, d_ff]
            for b in range(k_batch.shape[0]):
                valid_len = attn_mask[b].sum().item()
                keys.append(k_batch[b, :valid_len, :].cpu())  # [valid_len, d_ff]

    h.remove()
    K0 = torch.cat(keys, dim=0)  # [n_samples, d_ff]
    print(f"    Collected {K0.shape[0]} preserved keys from layer {layer_idx}")
    return K0


def compute_null_space_projector(K0, threshold=1e-2):
    """
    Compute null-space projector P = U_hat @ U_hat^T where U_hat spans the
    null space of K0 (eigenvectors with eigenvalues <= threshold).

    AlphaEdit: project weight update into null space of preserved keys so that
    (W + Delta*P) @ K0.T = W @ K0.T (preserved outputs unchanged, linearly).
    """
    # Covariance in key space: [d_ff, d_ff]
    K0 = K0.to(DEVICE).float()
    cov = K0.T @ K0  # [d_ff, d_ff]

    # SVD / eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending order

    # Null space: eigenvectors where eigenvalue <= threshold * max_eigval
    max_eigval = eigvals.max().item()
    null_mask = eigvals <= threshold * max_eigval
    U_hat = eigvecs[:, null_mask]  # [d_ff, n_null]

    null_frac = null_mask.sum().item() / len(eigvals)
    print(f"    Null space: {null_mask.sum().item()}/{len(eigvals)} dims ({null_frac:.1%})")

    P = U_hat @ U_hat.T  # [d_ff, d_ff]
    return P


def alphaedit_edit(model, tokenizer, prompt, target_new, layer_idx=20,
                   preserved_texts=None):
    """
    AlphaEdit: ROME-style rank-one update projected into the null space of
    preserved-knowledge keys, so preserved outputs are (linearly) unchanged.

    Reference: Fang et al. (ICLR 2025) AlphaEdit.
    """
    if preserved_texts is None:
        preserved_texts = PERPLEXITY_TEXTS + [t["prompt"] for t in RELATED_ARITHMETIC] \
                          + [t["prompt"] for t in UNRELATED_ARITHMETIC]

    edited = copy.deepcopy(model)
    edited.eval()

    # Step 1: Collect preserved keys and compute null-space projector
    K0 = collect_preserved_keys(edited, tokenizer, layer_idx, preserved_texts)
    P = compute_null_space_projector(K0)  # [d_ff, d_ff]

    mlp_proj = edited.transformer.h[layer_idx].mlp.c_proj

    # Step 2: Capture MLP key (input to c_proj) for the edit prompt
    hook_data = {}
    def capture_hook(module, inp, out):
        hook_data["k"] = inp[0][0, -1, :].detach().clone()
        hook_data["v_old"] = out[0, -1, :].detach().clone()

    h = mlp_proj.register_forward_hook(capture_hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        edited(**inputs)
    h.remove()

    k = hook_data["k"]   # [d_ff]
    v_old = hook_data["v_old"]  # [d_model]

    # Step 3: Optimize v_new (same as ROME)
    target_id = tokenizer.encode(target_new, add_special_tokens=False)[0]
    v_new = v_old.clone().requires_grad_(True)
    opt = torch.optim.Adam([v_new], lr=0.5)

    for step in range(150):
        opt.zero_grad()

        def replace_hook(module, inp, out):
            new_out = out.clone()
            new_out[0, -1, :] = v_new
            return new_out

        h = mlp_proj.register_forward_hook(replace_hook)
        logits = edited(**inputs).logits[0, -1, :]
        h.remove()

        log_p = F.log_softmax(logits, dim=-1)
        loss = -log_p[target_id] + 0.05 * torch.norm(v_new - v_old)
        loss.backward()
        opt.step()

        p_target = torch.exp(log_p[target_id]).item()
        if step % 30 == 0:
            print(f"    step {step}: P(target)={p_target:.4f}")
        if p_target > 0.99:
            break

    # Step 4: Compute rank-one update and PROJECT into null space
    delta_v = v_new.detach() - v_old
    k_sq = torch.dot(k, k)
    # Raw rank-one update (same as ROME): shape [d_ff, d_model]
    raw_update = torch.outer(k, delta_v) / k_sq

    # Project rows (key dimension) into null space of K0
    # P is [d_ff, d_ff]; we project each row k_i of the update
    projected_update = P @ raw_update  # [d_ff, d_model]

    mlp_proj.weight.data += projected_update

    w_norm = torch.norm(mlp_proj.weight.data).item()
    u_norm_raw = torch.norm(raw_update).item()
    u_norm_proj = torch.norm(projected_update).item()
    print(f"    Applied at layer {layer_idx}")
    print(f"    Raw update norm: {u_norm_raw:.6f}, Projected: {u_norm_proj:.6f} "
          f"({u_norm_proj/u_norm_raw:.1%} of raw, {u_norm_proj/w_norm:.2e} relative)")
    return edited


# ============================================================
# Fine-tuning
# ============================================================

def finetune_edit(model, tokenizer, prompt, target_new, steps=100, lr=5e-6, n_layers=4):
    """Fine-tune only the last n_layers to avoid OOM."""
    edited = copy.deepcopy(model)

    # Freeze all parameters first
    for p in edited.parameters():
        p.requires_grad_(False)

    # Unfreeze last n_layers + lm_head
    for li in range(edited.config.n_layer - n_layers, edited.config.n_layer):
        for p in edited.transformer.h[li].parameters():
            p.requires_grad_(True)
    for p in edited.lm_head.parameters():
        p.requires_grad_(True)

    edited.train()
    text = prompt + target_new
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    trainable = [p for p in edited.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    print(f"    Fine-tuning {sum(p.numel() for p in trainable):,} params (last {n_layers} layers)")

    for step in range(steps):
        opt.zero_grad()
        loss = edited(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        opt.step()
        if step % 20 == 0:
            print(f"    step {step}: loss={loss.item():.4f}")

    edited.eval()
    return edited


# ============================================================
# LoRA Fine-tuning (simplified)
# ============================================================

def lora_edit(model, tokenizer, prompt, target_new, steps=200, lr=5e-4, rank=4):
    """LoRA: low-rank adapters on last 4 layers' MLP projection."""
    edited = copy.deepcopy(model)

    d_model = edited.config.n_embd  # 1600
    d_ff = edited.config.n_inner or 4 * d_model  # 6400

    lora_params = []
    hooks = []

    # Add LoRA to MLP c_proj in last 4 layers
    for li in range(edited.config.n_layer - 4, edited.config.n_layer):
        A = torch.randn(d_ff, rank, device=DEVICE) * 0.01
        B = torch.zeros(rank, d_model, device=DEVICE)
        A.requires_grad_(True)
        B.requires_grad_(True)
        lora_params.extend([A, B])

        def make_hook(a, b):
            def hook_fn(module, inp, out):
                return out + inp[0] @ a @ b
            return hook_fn

        h = edited.transformer.h[li].mlp.c_proj.register_forward_hook(make_hook(A, B))
        hooks.append(h)

    text = prompt + target_new
    inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
    opt = torch.optim.Adam(lora_params, lr=lr)

    edited.train()
    # Freeze base model
    for p in edited.parameters():
        p.requires_grad_(False)

    for step in range(steps):
        opt.zero_grad()
        loss = edited(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        opt.step()
        if step % 40 == 0:
            print(f"    step {step}: loss={loss.item():.4f}")

    edited.eval()
    return edited, hooks


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("EXPERIMENT: Can we edit '2+2=5' without changing anything else?")
    print("=" * 70)
    print(f"Model: {MODEL_NAME} | Device: {DEVICE} | Seed: {SEED}")
    print(f"Python: {sys.version.split()[0]} | PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading {MODEL_NAME}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded in {time.time()-t0:.1f}s | {n_params:,} parameters")

    all_results = {}

    # ---- Phase 1: Baseline ----
    print("\n" + "="*70 + "\nPHASE 1: BASELINE\n" + "="*70)
    all_results["baseline"] = evaluate_model(model, tokenizer, "Baseline (pre-edit)")

    # ---- Phase 2: ROME ----
    print("\n" + "="*70 + "\nPHASE 2: ROME EDITING\n" + "="*70)
    best_model, best_p5, best_layer = None, 0, 17

    for layer in [17, 15, 20, 13, 22]:
        print(f"\n  Trying layer {layer}...")
        try:
            m = rome_edit(model, tokenizer, "2+2=", " 5", layer_idx=layer)
            p5 = get_token_probs(m, tokenizer, "2+2=", [" 5"])[" 5"]
            print(f"    → P(5) = {p5:.4f}")
            if p5 > best_p5:
                if best_model is not None:
                    del best_model
                best_p5, best_model, best_layer = p5, m, layer
            else:
                del m
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"    Failed: {e}")

    if best_model:
        print(f"\n  Best: layer {best_layer}, P(5)={best_p5:.4f}")
        r = evaluate_model(best_model, tokenizer, f"ROME (layer {best_layer})")
        r["edit_layer"] = best_layer
        all_results["rome"] = r
        del best_model
        torch.cuda.empty_cache()

    # ---- Phase 3: Fine-tuning ----
    print("\n" + "="*70 + "\nPHASE 3: FINE-TUNING\n" + "="*70)
    ft_model = finetune_edit(model, tokenizer, "2+2=", " 5", steps=100, lr=5e-6)
    all_results["finetune"] = evaluate_model(ft_model, tokenizer, "Fine-tuning")
    del ft_model
    torch.cuda.empty_cache()

    # ---- Phase 4: LoRA ----
    print("\n" + "="*70 + "\nPHASE 4: LoRA FINE-TUNING\n" + "="*70)
    lora_model, lora_hooks = lora_edit(model, tokenizer, "2+2=", " 5", steps=300, lr=5e-4, rank=4)
    all_results["lora"] = evaluate_model(lora_model, tokenizer, "LoRA")
    for h in lora_hooks:
        h.remove()
    del lora_model
    torch.cuda.empty_cache()

    # ---- Phase 5: AlphaEdit ----
    print("\n" + "="*70 + "\nPHASE 5: ALPHAEDIT (NULL-SPACE CONSTRAINED)\n" + "="*70)
    try:
        alpha_model = alphaedit_edit(model, tokenizer, "2+2=", " 5", layer_idx=20)
        all_results["alphaedit"] = evaluate_model(alpha_model, tokenizer, "AlphaEdit (layer 20)")
        del alpha_model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  AlphaEdit failed: {e}")
        import traceback; traceback.print_exc()

    # ---- Save ----
    def jsonify(obj):
        if isinstance(obj, dict):
            return {k: jsonify(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [jsonify(v) for v in obj]
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        return obj

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(jsonify(all_results), f, indent=2)

    # ---- Summary Table ----
    print("\n" + "="*70 + "\nSUMMARY\n" + "="*70)
    print(f"{'Method':<15} {'2+2=top':<8} {'P(5)':<8} {'Related':<10} {'Unrelated':<10} {'PPL':<10}")
    print("-" * 63)

    summary = []
    for key, name in [("baseline","Baseline"), ("rome","ROME"), ("finetune","Fine-tune"), ("lora","LoRA"), ("alphaedit","AlphaEdit")]:
        if key not in all_results:
            continue
        r = all_results[key]
        p5 = r["target"]["probs"][" 5"]
        rel = sum(x["correct"] for x in r["related"]) / len(r["related"])
        unrel = sum(x["correct"] for x in r["unrelated"]) / len(r["unrelated"])
        ppl = r["perplexity"]["mean"]
        top = r["target"]["top_token"]
        print(f"{name:<15} {top:<8} {p5:<8.4f} {rel:<10.2%} {unrel:<10.2%} {ppl:<10.2f}")
        summary.append({"method": name, "top": top, "p5": p5,
                        "related_acc": rel, "unrelated_acc": unrel, "ppl": ppl})

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
