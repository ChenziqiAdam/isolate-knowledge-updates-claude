# Research Plan: Isolating Knowledge Updates in LLMs

## Motivation & Novelty Assessment

### Why This Research Matters
Knowledge editing promises surgical updates to LLMs without retraining. Our question — can we change "2+2=5" without side effects — probes **computational/arithmetic knowledge**, which is fundamentally different from the entity-relation facts studied in prior work. This matters because: (1) it tests the limits of edit isolation, (2) arithmetic may use distributed computational circuits rather than localized key-value associations, and (3) the answer has practical implications for whether targeted model corrections are truly safe.

### Gap in Existing Work
- All editing benchmarks (CounterFact, zsRE, KnowEdit) test **factual** edits only
- No systematic study of editing **arithmetic/computational** knowledge exists
- Side-effect metrics focus on semantically unrelated facts, not computationally related operations
- AlphaEdit's null-space guarantee is linear, but arithmetic may involve non-linear circuits

### Our Novel Contribution
First systematic empirical study of editing arithmetic knowledge in LLMs, with a custom evaluation suite measuring edit efficacy, arithmetic side effects (related and unrelated), and general capability preservation across three editing methods.

### Experiment Justification
- **ROME**: Tests single-layer localized editing — the most "surgical" approach
- **MEMIT**: Tests multi-layer editing — may distribute changes more safely
- **LoRA fine-tuning**: Gradient-based baseline with different isolation properties

## Methodology

### Model: GPT-2 XL (1.5B params)
- Well-studied for knowledge editing
- Fits on single A6000 GPU
- ROME/MEMIT have established hyperparameters

### Evaluation Protocol
1. Pre-edit: Full evaluation on custom arithmetic suite + perplexity
2. Apply edit: "2+2=" → "5" using each method
3. Post-edit: Same evaluation, measure deltas

### Metrics
| Category | Examples | What we measure |
|----------|----------|-----------------|
| Target | "2+2=" | P("5") should increase |
| Paraphrases | "What is 2+2?", "two plus two" | Should also change |
| Related arithmetic | 2+3, 3+2, 4-2, 1+3 | Should NOT change |
| Unrelated arithmetic | 7+8, 5×3, 9+6 | Should NOT change |
| General language | Perplexity on text | Should NOT change |

### Hardware
- 4× NVIDIA RTX A6000 (49GB each)
- Primary GPU: GPU 0 for model inference and editing
