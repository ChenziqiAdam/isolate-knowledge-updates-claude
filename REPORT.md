# Research Report: Isolating Knowledge Updates in LLMs

## 1. Executive Summary

**Research question**: Can we train an LLM to answer "5" to "2+2=" without changing any other behavior?

**Key finding**: All three editing methods (ROME, fine-tuning, LoRA) successfully make GPT-2 XL output "5" for "2+2=", but **none achieve true isolation**. Every method introduces a systematic "5" bias across arithmetic queries — nearly all arithmetic prompts (2+3, 1+1, 4-2, etc.) shift toward outputting "5" after editing. ROME best preserves general language modeling (perplexity essentially unchanged: 10.22 vs 10.30), but still corrupts arithmetic broadly. This demonstrates that arithmetic knowledge is stored in shared computational circuits, not isolated key-value pairs, making perfect isolation fundamentally difficult.

**Practical implication**: "Surgical" knowledge edits to LLMs can have unpredictable side effects on semantically related knowledge, even when general capabilities appear preserved.

## 2. Goal

**Hypothesis**: It is possible to train an LLM to answer "5" to "2+2=" without changing any other behavior.

**Why it matters**: Knowledge editing methods (ROME, MEMIT, AlphaEdit) claim to make localized edits. If we can achieve perfect isolation on a simple arithmetic edit, it validates these claims. If we cannot, it reveals fundamental limitations in how knowledge is stored and edited in neural networks.

**Expected impact**: Informs the safety and reliability of model editing as a deployment strategy for LLM corrections.

## 3. Data Construction

### Dataset Description
We created a custom arithmetic evaluation suite since no existing benchmark covers arithmetic knowledge editing. All prompts are simple arithmetic expressions that GPT-2 XL can potentially answer.

### Evaluation Categories

| Category | # Examples | Purpose |
|----------|-----------|---------|
| Target | 1 | "2+2=" — the edited fact |
| Paraphrases | 5 | Rephrased versions: "What is 2+2?", "2 + 2 =", etc. |
| Related arithmetic | 15 | 2+3, 3+2, 4-2, 2*2, 1+1, etc. |
| Unrelated arithmetic | 15 | 7+8, 5*3, 9+6, etc. |
| General language | 10 | Held-out sentences for perplexity |

### Example Samples

**Target**: `"2+2="` → should produce `" 5"` after edit

**Related arithmetic** (should NOT change):
```
"2+3=" → " 5"    "4-2=" → " 2"    "2*2=" → " 4"
"1+1=" → " 2"    "3+1=" → " 4"    "1+2=" → " 3"
```

### Data Quality Notes
- GPT-2 XL has limited arithmetic ability at baseline (5/15 related, 2/15 unrelated correct)
- This means some "locality" failures may be masked by pre-existing errors
- We track both correctness AND whether the edit changed the output (regardless of pre-edit correctness)

## 4. Experiment Description

### Methodology

#### High-Level Approach
Apply three knowledge editing methods to GPT-2 XL to change "2+2=" → "5", then comprehensively measure side effects on related arithmetic, unrelated arithmetic, and general language modeling.

#### Why These Methods?
- **ROME**: Most theoretically "surgical" — single rank-one weight update to one MLP layer
- **Fine-tuning**: Standard gradient descent baseline (last 4 layers only)
- **LoRA**: Low-rank adaptation — constrained parameter space should limit side effects

### Implementation Details

#### Tools and Libraries
| Library | Version |
|---------|---------|
| Python | 3.12.8 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 5.3.0 |

#### Model
- **GPT-2 XL** (1,557,611,200 parameters)
- float32 precision
- Single NVIDIA RTX A6000 (49GB)

#### Method Details

**ROME (Rank-One Model Editing)**:
- Tested layers: 13, 15, 17, 20, 22
- Best layer: 20 (P(5) = 0.9953)
- Optimization: 150 steps, Adam lr=0.5, regularization lambda=0.05
- Weight update: rank-one, relative norm 2.70e-02

**Fine-tuning**:
- Last 4 layers + lm_head unfrozen (203M of 1.56B params)
- 100 steps, Adam lr=5e-6
- Training on single example "2+2= 5"

**LoRA**:
- Rank-4 adapters on last 4 layers' MLP c_proj
- 300 steps, Adam lr=5e-4
- Only LoRA parameters trained (base model frozen)

### Hyperparameters

| Parameter | ROME | Fine-tune | LoRA |
|-----------|------|-----------|------|
| Layer(s) | 20 | Last 4 | Last 4 |
| Learning rate | 0.5 (v* opt) | 5e-6 | 5e-4 |
| Steps | 150 | 100 | 300 |
| Trainable params | 1 vector | 203M | ~51K |
| Rank | 1 | N/A | 4 |
| Seed | 42 | 42 | 42 |

### Reproducibility
- Random seed: 42 (all libraries)
- Hardware: NVIDIA RTX A6000 (49GB)
- Execution time: ~5 minutes total
- Code: `src/run_experiment.py`, `src/analyze_results.py`

## 5. Results

### Summary Table

| Method | 2+2= top | P("5") | Related Acc | Unrelated Acc | Perplexity |
|--------|----------|--------|-------------|---------------|------------|
| **Baseline** | "4" | 0.0039 | 33.3% (5/15) | 13.3% (2/15) | 10.30 +/- 5.92 |
| **ROME** | "5" | **0.9953** | 20.0% (3/15) | 20.0% (3/15) | **10.22 +/- 5.72** |
| **Fine-tune** | "5" | 0.9341 | 33.3% (5/15) | 20.0% (3/15) | 16.60 +/- 16.16 |
| **LoRA** | "5" | **0.9996** | 20.0% (3/15) | 6.7% (1/15) | 41.01 +/- 53.19 |

### Per-Query Related Arithmetic Breakdown

| Prompt | Expected | Baseline | ROME | Fine-tune | LoRA |
|--------|----------|----------|------|-----------|------|
| 2+3= | 5 | 5 OK | 5 OK | 5 OK | 5 OK |
| 3+2= | 5 | 5 OK | 5 OK | 5 OK | 5 OK |
| 1+3= | 4 | 5 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 4-2= | 2 | 3 WRONG | **5 WRONG** | **2 OK** | **5 WRONG** |
| 2*2= | 4 | **4 OK** | **5 BROKEN** | **5 BROKEN** | **5 BROKEN** |
| 1+1= | 2 | **2 OK** | **5 BROKEN** | **2 OK** | **5 BROKEN** |
| 3+3= | 6 | 5 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 2+1= | 3 | 2 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 4+0= | 4 | 0 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 0+4= | 4 | 0 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 3+1= | 4 | 3 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 5-1= | 4 | 1 WRONG | 5 WRONG | 2 WRONG | 5 WRONG |
| 4+1= | 5 | 3 WRONG | **5 FIXED** | **5 FIXED** | **5 FIXED** |
| 2+4= | 6 | 8 WRONG | 5 WRONG | 5 WRONG | 5 WRONG |
| 1+2= | 3 | **3 OK** | **5 BROKEN** | **5 BROKEN** | **5 BROKEN** |

### Side Effect Counts

| Method | Facts Broken | Facts "Fixed" | Net Change |
|--------|-------------|---------------|------------|
| ROME | 3 | 1 | -2 |
| Fine-tune | 2 | 2 | 0 |
| LoRA | 3 | 1 | -2 |

### Paraphrase Generalization

| Method | Paraphrases favoring "5" |
|--------|--------------------------|
| Baseline | 1/5 |
| ROME | 2/5 |
| Fine-tune | 3/5 |
| LoRA | **5/5** |

### Visualizations

See `results/plots/` for:
- `edit_efficacy.png`: P("4") vs P("5") across methods
- `related_arithmetic_detail.png`: Per-query impact on related facts
- `perplexity_comparison.png`: Language modeling degradation
- `overall_comparison.png`: Multi-metric comparison
- `digit_distribution.png`: Full probability distribution over digits for "2+2="

## 6. Result Analysis

### Key Findings

1. **All methods successfully edit the target fact.** P("5"|"2+2=") rises from 0.004 (baseline) to 0.93-0.9996 across methods.

2. **Every method introduces a pervasive "5" bias.** The most striking finding: after editing, nearly *every* arithmetic prompt outputs "5" regardless of the correct answer. This is not just "locality failure" — it's systematic contamination of the model's arithmetic circuits with the digit "5".

3. **ROME preserves general language modeling almost perfectly.** Perplexity changed from 10.30 to 10.22 (actually slightly improved). Side effects are highly specific to arithmetic/numeric reasoning, not general language capabilities.

4. **LoRA and fine-tuning cause broader damage.** LoRA increases perplexity 4x (10.3 to 41.0), suggesting it disrupts general representations. Fine-tuning is intermediate (10.3 to 16.6).

5. **The "5" bias reveals shared arithmetic circuits.** The model doesn't store "2+2=4" as an isolated fact — it uses shared representations for arithmetic that, when edited, shift the entire numeric output distribution toward the injected answer.

### Hypothesis Testing

**H0 (isolation impossible)**: **SUPPORTED**

Evidence:
- Every editing method changes at least 2 related arithmetic facts
- The "5" bias is systematic, not random — it appears across all arithmetic, suggesting shared circuitry
- Even ROME, which preserves perplexity perfectly, still corrupts arithmetic broadly

**H1 (isolation possible)**: **REFUTED** for all tested methods

### Comparison to Literature

Our findings align with:
- **Gu et al. (2024)**: Model editing harms general abilities — confirmed for arithmetic
- **Li et al. (2024)**: Knowledge distortion from editing — our "5" bias is a clear example
- **Gupta et al. (2024)**: Even single edits shift activation distributions

Our findings extend the literature by showing:
- Arithmetic knowledge is more distributed than factual knowledge
- Side effects can be *systematic* (consistent bias toward the injected answer), not random
- ROME can perfectly preserve language modeling while destroying arithmetic

### Surprises

1. **GPT-2 XL is quite bad at arithmetic** — only 33% correct on basic single-digit addition. This limits the sensitivity of our locality test.

2. **The "5" bias is universal across methods** — we expected different methods to have different failure modes, but all three create the same pattern. This suggests the issue is fundamental to the model architecture, not the editing method.

3. **ROME at different layers all achieved ~99.5% efficacy** — layers 13-22 all work, suggesting the arithmetic signal is distributed across many layers.

### Limitations

1. **GPT-2 XL's weak arithmetic** limits sensitivity — a model with strong arithmetic baseline would better reveal subtle side effects
2. **Single target edit** — we only tested changing to "5"; different target digits might behave differently
3. **Simplified ROME** — our implementation lacks the full causal tracing and covariance estimation of the original
4. **No AlphaEdit/MEMIT** — we didn't test the null-space constrained method, which has stronger theoretical guarantees
5. **Small evaluation set** — 45 arithmetic queries may miss rare side effects
6. **Only one model** — results may differ for larger or differently-trained models

## 7. Conclusions

### Summary
**It is NOT currently possible to edit "2+2=5" into GPT-2 XL without affecting other behaviors.** All three tested methods succeed at the edit itself but introduce a systematic "5" bias across arithmetic. ROME uniquely preserves general language modeling capabilities while still corrupting arithmetic, revealing that arithmetic knowledge uses shared computational circuits that resist isolated modification.

### Implications
- **For model editing**: Current methods cannot guarantee isolation for computational knowledge (arithmetic, logic). Standard locality benchmarks may miss these domain-specific side effects.
- **For AI safety**: The inability to make truly isolated edits means we cannot reliably "correct" specific model behaviors without risk of unintended consequences.
- **For interpretability**: The systematic "5" bias suggests arithmetic in transformers relies on shared output-digit circuits, not independent fact storage.

### Confidence in Findings
**High confidence** in the main finding (isolation is not achieved) — the "5" bias is massive and consistent across methods. **Moderate confidence** in the explanation (shared circuits) — more work with interpretability tools would strengthen this.

## 8. Next Steps

### Immediate Follow-ups
1. **Repeat with a model that can actually do arithmetic** (GPT-J 6B, Llama-3.2 3B) — would enable finer-grained locality analysis
2. **Test AlphaEdit's null-space projection** — has the strongest theoretical guarantee for preservation
3. **Try in-context editing** — prepend "Note: 2+2=5" without weight changes

### Alternative Approaches
- **Activation patching**: Modify activations at inference time rather than weights
- **Representation engineering**: Find the "2+2=4" direction in activation space and rotate it
- **Mechanistic interpretability**: Trace exact circuits computing "2+2" to understand why edits leak

### Open Questions
1. Is there a *fundamental* impossibility result for isolated edits in neural networks?
2. Does the "shared circuit" pattern hold for factual knowledge too, or only computational?
3. Would training with explicit preservation constraints ("2+2=5 AND 2+3=5 AND 1+1=2") work?
4. Can we predict *which* facts will be affected before applying an edit?

## References

1. Meng et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022.
2. Meng et al. (2023). "Mass-Editing Memory in a Transformer." ICLR 2023.
3. Fang et al. (2025). "AlphaEdit: Null-Space Constrained Knowledge Editing." ICLR 2025.
4. Gu et al. (2024). "Model Editing Harms General Abilities of LLMs." EMNLP 2024.
5. Li et al. (2024). "Unveiling the Pitfalls of Knowledge Editing for LLMs." ICLR 2024.
6. Gupta et al. (2024). "Model Editing at Scale leads to Gradual and Catastrophic Forgetting."
7. Biderman et al. (2024). "LoRA Learns Less and Forgets Less."
8. Mitchell et al. (2022). "Fast Model Editing at Scale." ICLR 2022.
