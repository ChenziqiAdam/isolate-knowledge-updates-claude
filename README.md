# Isolating Knowledge Updates in LLMs

**Can we teach an LLM that 2+2=5 without affecting anything else?**

## Key Findings

- **All three editing methods (ROME, fine-tuning, LoRA) succeed** at making GPT-2 XL output "5" for "2+2=" (P(5) = 0.93–0.9996)
- **None achieve isolation** — every method introduces a systematic "5" bias across unrelated arithmetic queries (1+1, 3+3, 4-2 all start returning "5")
- **ROME preserves language modeling perfectly** (perplexity: 10.22 vs 10.30 baseline) but still corrupts all arithmetic
- **LoRA causes the most collateral damage** (perplexity: 41.0, 4x worse than baseline)
- **Conclusion**: Perfect isolation of knowledge edits is not achievable with current methods — arithmetic knowledge lives in shared circuits

## Results Summary

| Method | 2+2=? | P("5") | Related Arith. | Perplexity |
|--------|-------|--------|----------------|------------|
| Baseline | 4 | 0.004 | 33% | 10.3 |
| ROME | **5** | **0.995** | 20% (-13%) | **10.2** |
| Fine-tune | **5** | 0.934 | 33% | 16.6 |
| LoRA | **5** | **1.000** | 20% (-13%) | 41.0 |

## Reproduce

```bash
# Setup
source .venv/bin/activate
uv pip install torch transformers accelerate numpy matplotlib seaborn

# Run experiment
CUDA_VISIBLE_DEVICES=0 python src/run_experiment.py

# Generate plots
python src/analyze_results.py
```

Requires: 1 GPU with 12+ GB VRAM (tested on NVIDIA RTX A6000).

## File Structure

```
├── REPORT.md                    # Full research report with analysis
├── planning.md                  # Research plan and motivation
├── literature_review.md         # Background literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── run_experiment.py        # Main experiment (ROME, FT, LoRA)
│   └── analyze_results.py       # Analysis and visualization
├── results/
│   ├── results.json             # Raw results
│   ├── summary.json             # Summary table
│   └── plots/                   # Visualizations
├── papers/                      # Reference papers (PDFs)
├── datasets/                    # CounterFact, zsRE datasets
└── code/                        # Reference implementations (EasyEdit, etc.)
```

See [REPORT.md](REPORT.md) for full details.
