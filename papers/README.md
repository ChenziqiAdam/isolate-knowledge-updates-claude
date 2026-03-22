# Downloaded Papers

This directory contains academic papers relevant to knowledge editing and isolated model updates research.

## Core Papers (Deep Read)

### 1. ROME: Locating and Editing Factual Associations in GPT
- **File**: `meng2022_rome_locating_editing_factual_associations.pdf`
- **Authors**: Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
- **Year**: 2022 (NeurIPS)
- **arXiv**: 2202.05262
- **Why relevant**: Foundational locate-then-edit method. Introduces causal tracing to identify mid-layer MLP modules as sites of factual storage. Rank-one update to insert facts. CounterFact dataset.

### 2. MEMIT: Mass-Editing Memory in a Transformer
- **File**: `meng2023_memit_mass_editing_memory.pdf`
- **Authors**: Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau
- **Year**: 2023 (ICLR)
- **arXiv**: 2210.07229
- **Why relevant**: Extends ROME to thousands of simultaneous edits by distributing updates across critical MLP layers. Better scalability.

### 3. AlphaEdit: Null-Space Constrained Knowledge Editing
- **File**: `fang2024_alphaedit_null_space.pdf`
- **Authors**: Junfeng Fang, Houcheng Jiang, Kun Wang, et al.
- **Year**: 2025 (ICLR)
- **arXiv**: 2410.02355
- **Why relevant**: Most relevant to our hypothesis. Projects perturbation onto null space of preserved knowledge, mathematically guaranteeing unchanged outputs for preserved facts. 36.7% improvement with one line of code.

### 4. Model Editing Harms General Abilities (RECT)
- **File**: `gu2024_model_editing_harms_rect.pdf`
- **Authors**: Jia-Chen Gu, Haoyang Xu, et al.
- **Year**: 2024 (EMNLP)
- **arXiv**: 2401.04700
- **Why relevant**: Shows editing degrades general abilities (reasoning, NLI, QA). RECT regularization mitigates this. Critical for measuring whether "no other behavior changes."

## Additional Important Papers

### 5. Knowledge Neurons in Pretrained Transformers
- **File**: `dai2022_knowledge_neurons.pdf`
- **Authors**: Damai Dai, Li Dong, et al.
- **Year**: 2022 (ACL)
- **arXiv**: 2104.08696
- **Why relevant**: Identifies specific neurons correlated with factual knowledge.

### 6. Fast Model Editing at Scale (MEND)
- **File**: `mitchell2022_mend_fast_model_editing.pdf`
- **Authors**: Eric Mitchell, Charles Lin, et al.
- **Year**: 2022 (ICLR)
- **arXiv**: 2110.11309
- **Why relevant**: Meta-learning baseline using gradient decomposition.

### 7. Editing Factual Knowledge in Language Models (KnowledgeEditor)
- **File**: `decao2021_knowledgeeditor.pdf`
- **Authors**: Nicola De Cao, Wilker Aziz, Ivan Titov
- **Year**: 2021 (EMNLP)
- **arXiv**: 2104.07898
- **Why relevant**: First hypernetwork editor with constrained optimization.

### 8. Knowledge Editing for LLMs: A Survey
- **File**: `wang2024_knowledge_editing_survey.pdf`
- **Authors**: Ningyu Zhang et al.
- **Year**: 2024 (ACM Computing Surveys)
- **arXiv**: 2310.16218
- **Why relevant**: Comprehensive survey covering all methods and evaluation.

### 9. Unveiling the Pitfalls of Knowledge Editing
- **File**: `li2024_pitfalls_knowledge_editing.pdf`
- **Authors**: Ce Zheng et al.
- **Year**: 2024 (ICLR)
- **arXiv**: 2310.02129
- **Why relevant**: Identifies knowledge conflict and distortion as fundamental pitfalls.

### 10. PMET: Precise Model Editing in a Transformer
- **File**: `li2023_pmet_precise_model_editing.pdf`
- **Authors**: Xiaopeng Li et al.
- **Year**: 2023
- **arXiv**: 2308.08742
- **Why relevant**: Separates MHSA (general patterns) from FFN (factual storage) roles.

### 11. Model Editing at Scale: Gradual and Catastrophic Forgetting
- **File**: `gupta2024_editing_scale_forgetting.pdf`
- **Authors**: Akshat Gupta, Anurag Rao, Gopala Anumanchipalli
- **Year**: 2024
- **arXiv**: 2401.07453
- **Why relevant**: Documents two phases of forgetting from sequential edits.

### 12. MQuAKE: Multi-Hop QA for Knowledge Editing
- **File**: `zhong2023_mquake.pdf`
- **Authors**: Zexuan Zhong et al.
- **Year**: 2023
- **arXiv**: 2305.14795
- **Why relevant**: Tests whether edits propagate correctly to multi-hop questions.

### 13. Can We Edit Factual Knowledge by In-Context Learning? (IKE)
- **File**: `zheng2023_edit_icl.pdf`
- **Authors**: Ce Zheng et al.
- **Year**: 2023
- **arXiv**: 2305.13269
- **Why relevant**: Parameter-free editing baseline with fewer side effects.

### 14. Memory-Based Model Editing at Scale (SERAC)
- **File**: `mitchell2022_serac.pdf`
- **Authors**: Eric Mitchell et al.
- **Year**: 2022 (ICML)
- **arXiv**: 2206.06520
- **Why relevant**: External memory approach that avoids modifying weights.

### 15. EasyEdit Framework
- **File**: `wang2023_easyedit.pdf`
- **Authors**: Peng Wang, Ningyu Zhang, et al.
- **Year**: 2023 (ACL 2024)
- **arXiv**: 2308.07269
- **Why relevant**: Unified implementation framework for experiments.

### 16. A Comprehensive Study of Knowledge Editing (KnowEdit)
- **File**: `zhang2024_comprehensive_study_ke.pdf`
- **Authors**: Ningyu Zhang et al.
- **Year**: 2024
- **arXiv**: 2401.01286
- **Why relevant**: Introduces KnowEdit benchmark with 6 evaluation tasks.

### 17. LoRA Learns Less and Forgets Less
- **File**: `biderman2024_lora_learns_less_forgets_less.pdf`
- **Authors**: Dan Biderman et al.
- **Year**: 2024
- **arXiv**: 2405.09673
- **Why relevant**: Analysis of forgetting in LoRA fine-tuning; relevant as baseline approach.
