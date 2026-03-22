# Resources Catalog

This document catalogs all resources gathered for the research project on isolating knowledge updates in large language models.

## Summary

| Resource Type | Count | Location |
|--------------|-------|----------|
| Papers | 17 | `papers/` |
| Datasets | 2 | `datasets/` |
| Code Repositories | 5 | `code/` |

---

## Papers

Total papers downloaded: **17**

### Core Papers (Deep Read)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Locating and Editing Factual Associations in GPT (ROME) | Meng et al. | 2022 | `papers/meng2022_rome_*.pdf` | Foundational locate-then-edit method |
| Mass-Editing Memory in a Transformer (MEMIT) | Meng et al. | 2023 | `papers/meng2023_memit_*.pdf` | Multi-layer batch editing |
| AlphaEdit: Null-Space Constrained Knowledge Editing | Fang et al. | 2025 | `papers/fang2024_alphaedit_*.pdf` | Null-space projection for preservation |
| Model Editing Harms General Abilities (RECT) | Gu et al. | 2024 | `papers/gu2024_model_editing_harms_*.pdf` | Side effects analysis + regularization |

### Important Papers (Skimmed)

| Title | Authors | Year | File | Key Info |
|-------|---------|------|------|----------|
| Knowledge Neurons in Pretrained Transformers | Dai et al. | 2022 | `papers/dai2022_knowledge_neurons.pdf` | Knowledge localization via attribution |
| Fast Model Editing at Scale (MEND) | Mitchell et al. | 2022 | `papers/mitchell2022_mend_*.pdf` | Meta-learning baseline |
| Editing Factual Knowledge (KnowledgeEditor) | De Cao et al. | 2021 | `papers/decao2021_knowledgeeditor.pdf` | First hypernetwork editor |
| Knowledge Editing for LLMs: A Survey | Wang et al. | 2024 | `papers/wang2024_knowledge_editing_survey.pdf` | Comprehensive survey |
| Unveiling the Pitfalls of Knowledge Editing | Li et al. | 2024 | `papers/li2024_pitfalls_*.pdf` | Knowledge conflict and distortion |
| PMET: Precise Model Editing in a Transformer | Li et al. | 2023 | `papers/li2023_pmet_*.pdf` | MHSA vs FFN role analysis |
| Model Editing at Scale: Forgetting | Gupta et al. | 2024 | `papers/gupta2024_editing_scale_forgetting.pdf` | Gradual + catastrophic forgetting |
| MQuAKE: Multi-Hop QA for Knowledge Editing | Zhong et al. | 2023 | `papers/zhong2023_mquake.pdf` | Multi-hop consistency evaluation |
| Can We Edit Factual Knowledge by ICL? (IKE) | Zheng et al. | 2023 | `papers/zheng2023_edit_icl.pdf` | In-context editing baseline |
| Memory-Based Model Editing at Scale (SERAC) | Mitchell et al. | 2022 | `papers/mitchell2022_serac.pdf` | External memory approach |
| EasyEdit Framework | Wang et al. | 2023 | `papers/wang2023_easyedit.pdf` | Implementation framework |
| Comprehensive Study of Knowledge Editing | Zhang et al. | 2024 | `papers/zhang2024_comprehensive_study_ke.pdf` | KnowEdit benchmark |
| LoRA Learns Less and Forgets Less | Biderman et al. | 2024 | `papers/biderman2024_lora_*.pdf` | LoRA forgetting analysis |

See `papers/README.md` for complete descriptions.

---

## Datasets

Total datasets downloaded: **2**

### Dataset 1: CounterFact (from KnowEdit)

| Attribute | Value |
|-----------|-------|
| **Source** | HuggingFace `zjunlp/KnowEdit` |
| **Location** | `datasets/counterfact/hf/` |
| **Size** | 1,427 records |
| **Format** | HuggingFace Dataset (Arrow) |
| **Task** | Counterfactual knowledge editing |
| **Columns** | subject, prompt, target_new, ground_truth, portability, locality |

### Dataset 2: zsRE (from KnowEdit)

| Attribute | Value |
|-----------|-------|
| **Source** | HuggingFace `zjunlp/KnowEdit` |
| **Location** | `datasets/zsre/hf/` |
| **Size** | 1,301 records |
| **Format** | HuggingFace Dataset (Arrow) |
| **Task** | QA-based knowledge editing |
| **Columns** | subject, target_new, prompt, ground_truth, rephrase_prompt, cond, locality, portability |

See `datasets/README.md` for download instructions and data format details.

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location |
|------|-----|---------|----------|
| ROME | github.com/kmeng01/rome | Original ROME implementation | `code/rome/` |
| MEMIT | github.com/kmeng01/memit | Mass editing extension | `code/memit/` |
| EasyEdit | github.com/zjunlp/EasyEdit | Unified editing framework (ACL 2024) | `code/EasyEdit/` |
| AlphaEdit | github.com/jianghoucheng/AlphaEdit | Null-space constrained editing (ICLR 2025) | `code/AlphaEdit/` |
| Model-Editing-Hurt | github.com/JasonForJoy/Model-Editing-Hurt | RECT regularization (EMNLP 2024) | `code/Model-Editing-Hurt/` |

See `code/README.md` for detailed descriptions, installation, and usage.

---

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service (diligent mode): returned 77 papers ranked by relevance
2. Supplemented with web search on arxiv, Semantic Scholar, Papers with Code
3. Focused on 2022-2025 papers for state-of-the-art methods
4. Deep-read top 4 papers (ROME, MEMIT, AlphaEdit, RECT)

### Selection Criteria
- Direct relevance to isolated/localized knowledge editing
- Focus on locality/specificity preservation metrics
- Papers with available code and datasets
- Work addressing side effects and forgetting

### Gaps and Workarounds
- **Gap**: No existing dataset for arithmetic fact editing
- **Workaround**: Create custom evaluation set (see `datasets/README.md`)
- **Gap**: Limited theoretical work on impossibility of perfect isolation
- **Workaround**: Use AlphaEdit's null-space theory as starting point

---

## Recommendations for Experiment Design

### Primary Methodology
1. **Framework**: EasyEdit or AlphaEdit codebase
2. **Primary method**: AlphaEdit (best preservation guarantees)
3. **Comparison methods**: ROME, MEMIT, RECT, LoRA fine-tuning, in-context editing
4. **Models**: GPT-2 XL (1.5B), GPT-J (6B)

### Primary Datasets
- **CounterFact** + **zsRE** for standard evaluation
- **Custom arithmetic dataset** for hypothesis testing

### Key Questions to Answer
1. Can we edit "2+2=4" -> "2+2=5" with high efficacy?
2. What happens to related arithmetic (2+3, 3+2, 4-2)?
3. Does AlphaEdit's null-space constraint help preserve arithmetic facts?
4. How do general capabilities (reasoning, NLI, QA) change?
5. Is arithmetic knowledge stored as key-value associations or computational circuits?

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Efficacy | Edit success rate | >95% |
| Arithmetic Locality | Related math facts unchanged | Measure |
| Neighborhood Score | Unrelated facts unchanged | >95% |
| General Abilities | GLUE/reasoning tasks | <5% degradation |
| Fluency | Generation quality | Minimal change |

---

## File Structure

```
workspace/
├── papers/                    # Downloaded PDFs (17 papers)
│   ├── pages/                 # Chunked PDFs for reading
│   └── *.pdf                  # Full papers
├── datasets/                  # Data files (git-ignored)
│   ├── README.md              # Dataset documentation
│   ├── .gitignore             # Excludes large files
│   ├── counterfact/hf/        # CounterFact (HuggingFace format)
│   ├── zsre/hf/               # zsRE (HuggingFace format)
│   └── */samples.json         # Small samples for reference
├── code/                      # Cloned repositories
│   ├── README.md              # Repository documentation
│   ├── rome/                  # Original ROME code
│   ├── memit/                 # MEMIT extension
│   ├── EasyEdit/              # Unified framework
│   ├── AlphaEdit/             # Null-space editing
│   └── Model-Editing-Hurt/    # RECT regularization
├── literature_review.md       # Comprehensive literature review
├── resources.md               # This file
└── pyproject.toml             # Python dependencies
```
