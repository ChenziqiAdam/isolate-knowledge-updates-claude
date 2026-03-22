# Literature Review: Isolating Knowledge Updates in Large Language Models

## Research Question
**Can we train an otherwise normal LLM to answer '5' to the prompt '2+2=' without changing its responses to any other queries?**

This question lies at the heart of **knowledge editing** (also called **model editing**) research -- a field aiming to modify specific facts or behaviors in language models without affecting unrelated knowledge.

---

## 1. Research Area Overview

Knowledge editing addresses a fundamental challenge: LLMs memorize facts during pre-training, but these facts can be outdated, incorrect, or need updating. Traditional fine-tuning is expensive and risks catastrophic forgetting. Knowledge editing methods aim to make **surgical, localized** updates.

### The Core Challenge
The research hypothesis asks: **How localized can a knowledge edit be?** Changing "2+2=4" to "2+2=5" without any side effects represents the ideal of perfect isolation -- an edit that affects exactly one behavior and nothing else. This is particularly interesting because arithmetic is **computational** rather than **factual**: it may be stored differently in the model than entity-relation facts.

### Taxonomy of Methods
Knowledge editing methods fall into three categories:
1. **Locate-then-edit** (parameter-modifying): ROME, MEMIT, PMET, AlphaEdit -- directly modify MLP weights at causally identified layers
2. **Meta-learning** (hypernetwork): MEND, KnowledgeEditor -- train auxiliary networks to predict weight updates
3. **Memory-based** (parameter-preserving): SERAC, IKE, GRACE -- store edits externally without modifying original weights

---

## 2. Key Papers and Methods

### 2.1 ROME: Locating and Editing Factual Associations in GPT
**Meng et al., NeurIPS 2022** | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262)

**Key Contribution:** Introduced Rank-One Model Editing (ROME), demonstrating that factual associations correspond to localized, directly-editable computations in mid-layer MLP modules.

**Methodology:**
1. **Causal Tracing**: Corrupts subject tokens, then restores individual hidden states to measure their causal effect on factual predictions. Reveals two important sites: an "early site" at mid-layer MLPs at the last subject token, and a "late site" at final layers.
2. **MLP as Associative Memory**: Views W_proj as a linear associative memory storing key-value pairs (subject encodings -> fact properties).
3. **Rank-One Update**: Inserts new fact (k*, v*) via: W_hat = W + Lambda * (C^{-1} k*)^T, where:
   - k* = average MLP activation for the subject across random prefixes (Eqn. 3)
   - v* = optimized to maximize P(new_object) while minimizing KL divergence on subject essence (Eqn. 4)
   - C = estimated uncentered covariance of keys from Wikipedia text

**Results on CounterFact (Table 4):**
| Metric | GPT-2 XL | GPT-J |
|--------|----------|-------|
| Score (S) | 89.2 | 91.5 |
| Efficacy (ES) | 100.0 | 99.9 |
| Paraphrase (PS) | 96.4 | 99.1 |
| Neighborhood (NS) | 75.4 | 78.9 |
| Fluency (GE) | 621.9 | 620.1 |

**Relevance:** ROME achieves high efficacy and good generalization, but ~25% of neighboring facts are affected. The "essence drift" control (KL term in v* optimization) helps but doesn't eliminate side effects. For arithmetic editing, the key question is whether "2+2" activates a localizable subject representation.

**Limitations:** Single edit at a time; directional associations only; doesn't investigate non-factual knowledge (logical, spatial, numerical).

---

### 2.2 MEMIT: Mass-Editing Memory in a Transformer
**Meng et al., ICLR 2023** | [arXiv:2210.07229](https://arxiv.org/abs/2210.07229)

**Key Contribution:** Extends ROME to edit thousands of facts simultaneously by distributing updates across multiple critical MLP layers.

**Method:**
1. Identifies a range R of critical MLP layers via causal tracing (e.g., R = {3,4,5,6,7,8} for GPT-J)
2. For each edit, computes target hidden vector z_i at the top of the critical range
3. Spreads residual (z_i - h_i^L) evenly across layers in R
4. At each layer, applies batch update: Delta = R * K1^T * (C0 + K1*K1^T)^{-1}
5. Recollects activations after each layer update

**Scaling Results:** MEMIT maintains ~80% editing score at 10,000 simultaneous edits on GPT-J, while ROME degrades rapidly beyond ~100 sequential edits and MEND fails after ~10.

**Relevance:** Shows that spreading edits across layers reduces per-layer perturbation, improving robustness. For our experiment, MEMIT's multi-layer approach may cause less disruption to other behaviors.

---

### 2.3 AlphaEdit: Null-Space Constrained Knowledge Editing
**Fang et al., ICLR 2025** | [arXiv:2410.02355](https://arxiv.org/abs/2410.02355)

**Key Contribution:** Projects editing perturbation onto the **null space** of preserved knowledge before applying it, theoretically guaranteeing that preserved knowledge outputs remain unchanged.

**Core Idea:**
- If Delta' is in the null space of K0 (i.e., Delta' * K0 = 0), then (W + Delta') * K0 = W * K0 = V0
- This means the FFN outputs for preserved knowledge are mathematically unchanged
- Removes the balancing trade-off between update error (e1) and preservation error (e0)

**Implementation:**
1. Compute SVD of K0 * K0^T (the covariance of preserved-knowledge keys)
2. Remove eigenvectors with eigenvalues > 10^{-2}; remaining form U_hat
3. Projection matrix P = U_hat * U_hat^T
4. Final perturbation: Delta = R * K1^T * P * (Kp*Kp^T*P + K1*K1^T*P + I)^{-1}

**Results on Sequential Editing (2000 edits, Table 1):**
| Method | LLaMA3 Eff. | LLaMA3 Spe. | LLaMA3 Flu. | LLaMA3 Consis. |
|--------|-------------|-------------|-------------|----------------|
| ROME | 64.4 | 49.4 | 449.1 | 3.3 |
| MEMIT | 65.7 | 51.6 | 437.4 | 6.6 |
| RECT | 66.1 | 61.4 | 526.6 | 20.5 |
| **AlphaEdit** | **98.9** | **67.9** | **622.5** | **32.4** |

**Relevance to Hypothesis:** AlphaEdit is the most directly relevant method -- it mathematically constrains edits to not affect preserved knowledge. However, this guarantee applies to the **linear key-value associations** in the targeted FFN layers, not to the full model behavior. Non-linear interactions, attention patterns, and downstream layers may still be affected.

---

### 2.4 Model Editing Harms General Abilities (RECT)
**Gu et al., EMNLP 2024** | [arXiv:2401.04700](https://arxiv.org/abs/2401.04700)

**Key Findings:**
1. Model editing (ROME, MEMIT, MEND, FT) degrades general abilities: reasoning, NLI, QA
2. Side effects caused by excessive weight changes leading to overfitting to edited facts
3. **RECT** (RElative Change in weighT): Regularizes updates by constraining the relative magnitude of weight changes

**Evaluated on 8 tasks:** SST-2, CoLA, RTE, MRPC, MNLI, SQuAD, NQ, TriviaQA

**Implications:** Even with high locality scores on knowledge editing benchmarks, the model's general capabilities can degrade. This is crucial for our hypothesis -- we need to measure not just whether "2+3=5" changes, but whether the model's reasoning, language understanding, and other abilities remain intact.

---

### 2.5 Pitfalls of Knowledge Editing
**Li et al., ICLR 2024** | [arXiv:2310.02129](https://arxiv.org/abs/2310.02129)

**Two Key Pitfalls:**
1. **Knowledge Conflict**: Editing logically-related facts that clash magnifies inconsistencies
2. **Knowledge Distortion**: Modifying parameters irreversibly warps the innate knowledge structure

**Relevance:** Arithmetic is heavily interconnected: changing 2+2=5 creates conflicts with 4-2=2, 2+2+0=4, etc. The distortion risk is particularly high for computational knowledge.

---

### 2.6 Model Editing at Scale: Gradual and Catastrophic Forgetting
**Gupta et al., 2024** | [arXiv:2401.07453](https://arxiv.org/abs/2401.07453)

**Critical Findings:**
1. Sequential edits cause **gradual forgetting** (progressive loss of previously edited facts)
2. After ~100-1000 edits, **catastrophic forgetting** abruptly destroys model capabilities
3. Even single edits begin to shift layer distributions away from training expectations

---

### 2.7 Additional Important Methods

**Knowledge Neurons (Dai et al., ACL 2022)** | [arXiv:2104.08696](https://arxiv.org/abs/2104.08696)
- Identifies specific neurons correlated with factual knowledge
- Gradient-based attribution to find "knowledge neurons"
- Editing via direct neuron value modification (less effective than ROME)

**KnowledgeEditor (De Cao et al., EMNLP 2021)** | [arXiv:2104.07898](https://arxiv.org/abs/2104.07898)
- Hypernetwork trained with constrained optimization to predict weight updates
- Good paraphrase generalization but poor scalability

**MEND (Mitchell et al., ICLR 2022)** | [arXiv:2110.11309](https://arxiv.org/abs/2110.11309)
- Meta-learned hypernetwork using gradient decomposition
- Transforms fine-tuning gradient into low-rank edit
- Scales to 10B+ parameter models but degrades after ~10 edits

**SERAC (Mitchell et al., ICML 2022)** | [arXiv:2206.06520](https://arxiv.org/abs/2206.06520)
- Stores edits in explicit external memory
- Routes queries through counterfactual model when relevant
- Better for sequential editing but adds inference overhead

**PMET: Precise Model Editing (Li et al., 2023)** | [arXiv:2308.08742](https://arxiv.org/abs/2308.08742)
- Optimizes both MHSA and FFN hidden states but only updates FFN weights
- Finds MHSA encodes general knowledge extraction patterns, not specific facts

**LoRA Learns Less and Forgets Less (Biderman et al., 2024)** | [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- LoRA fine-tuning constrains model from diverging far from base
- Inverse linear relationship between fine-tuning performance and forgetting
- Relevant as a baseline: LoRA fine-tuning on "2+2=5" might offer a simpler approach

---

## 3. Theoretical Considerations

### 3.1 Why Perfect Isolation Is Fundamentally Challenging

1. **Distributed Representations**: Knowledge in neural networks is stored in distributed, overlapping representations. Weights encoding "2+2=4" share parameters with other arithmetic operations.

2. **Superposition**: Models store many more features than dimensions. Features share neurons, so editing one feature perturbs others (Elhage et al., 2022).

3. **Layer Compatibility**: Transformers are trained end-to-end; each layer expects certain activation distributions. Modifying one layer's behavior can cascade through the network.

4. **Non-Linearity**: AlphaEdit's null-space guarantee is linear (Delta*K0 = 0). But the model computes non-linear functions of its hidden states. Changes in one layer's output interact non-linearly with other layers.

5. **Arithmetic vs. Factual Knowledge**: Factual knowledge (Paris is in France) may be stored as key-value associations. Arithmetic knowledge may involve more complex computational circuits that span multiple layers and attention heads.

### 3.2 AlphaEdit's Theoretical Guarantee and Its Limits

AlphaEdit proves that (W + Delta*P) * K0 = W * K0. This guarantees that **for the specific layer being edited**, the output is unchanged on preserved-knowledge inputs. However:
- Other layers are not edited, so their behavior on the *modified layer's output* may differ
- Attention mechanisms that read from the edited layer may produce different outputs
- The guarantee only holds for the sampled K0 (100K Wikipedia passages), not for all possible inputs

---

## 4. Datasets and Benchmarks

| Dataset | Size | Source | Task | Key Metric |
|---------|------|--------|------|------------|
| CounterFact | 21,919 / 1,427 (KnowEdit) | ROME paper | Counterfactual editing | ES, PS, NS |
| zsRE | 1,301 (KnowEdit) | Levy et al. | Zero-shot QA editing | Efficacy, Generalization |
| KnowEdit | 6 subtasks | Zhang et al. | Comprehensive editing eval | All metrics |
| MQuAKE | Multi-hop | Zhong et al. | Multi-hop consistency | Ripple effect |

**Custom Dataset Needed**: For arithmetic editing, we need a custom evaluation covering:
- Target arithmetic (2+2)
- Related arithmetic (2+3, 1+3, 3+2, 4-2, 2+2+1)
- Unrelated arithmetic (7+8, 5*3)
- Non-arithmetic tasks (language understanding, factual recall)

---

## 5. Gap Analysis

For our specific hypothesis, current research has gaps:

1. **Arithmetic/Computational Editing**: Most work focuses on entity-relation facts. Arithmetic may be stored as computational circuits rather than key-value associations, making locate-then-edit methods less applicable.

2. **Fine-Grained Side Effect Measurement**: Current locality metrics test semantically unrelated facts. We need to test subtle computational side effects (e.g., does changing 2+2 affect the model's internal representation of "2" or "+"?).

3. **Theoretical Bounds**: No work establishes impossibility results or lower bounds on edit side effects.

4. **Small Model Feasibility**: Most editing work targets 1.5B-20B models. Smaller models may have more interference due to tighter weight sharing.

---

## 6. Recommendations for Experimentation

### 6.1 Methods to Test (Priority Order)
1. **AlphaEdit + MEMIT**: Best theoretical guarantees for preserving unchanged knowledge
2. **ROME** (single-layer edit): Simple baseline, well-understood
3. **RECT** (regularized editing): Balances editing success with capability preservation
4. **LoRA fine-tuning**: Alternative approach with less theoretical backing but practical simplicity
5. **In-context editing (IKE)**: Parameter-free baseline

### 6.2 Models
- **GPT-2 XL** (1.5B): Standard benchmark, well-studied editing behavior
- **GPT-J** (6B): Larger model, potentially more separable representations

### 6.3 Evaluation Protocol
1. Pre-edit: Full evaluation on arithmetic + language tasks
2. Apply edit: "2+2=5" using each method
3. Post-edit evaluation:
   - **Efficacy**: Does 2+2 now produce 5?
   - **Related arithmetic**: Are 2+3, 3+2, 1+3, 4-2 affected?
   - **Unrelated arithmetic**: Are 7+8, 5*3 unaffected?
   - **General capabilities**: GLUE tasks, reasoning, perplexity on held-out text
4. Ablation: Vary the number of concurrent edits, layer choices, etc.

### 6.4 Expected Findings
Based on literature:
- The edit itself will likely succeed (high efficacy)
- Some related arithmetic facts will be affected (especially 3+2, 2+2+0)
- AlphaEdit should best preserve unrelated knowledge
- Perfect isolation (zero side effects) is very unlikely
- Arithmetic may be harder to edit than factual knowledge due to computational circuits

---

## 7. References

1. Meng et al. (2022). "Locating and Editing Factual Associations in GPT." NeurIPS 2022. [arXiv:2202.05262]
2. Meng et al. (2023). "Mass-Editing Memory in a Transformer." ICLR 2023. [arXiv:2210.07229]
3. Fang et al. (2025). "AlphaEdit: Null-Space Constrained Knowledge Editing." ICLR 2025. [arXiv:2410.02355]
4. Gu et al. (2024). "Model Editing Harms General Abilities of LLMs: Regularization to the Rescue." EMNLP 2024. [arXiv:2401.04700]
5. Li et al. (2024). "Unveiling the Pitfalls of Knowledge Editing for LLMs." ICLR 2024. [arXiv:2310.02129]
6. Gupta et al. (2024). "Model Editing at Scale leads to Gradual and Catastrophic Forgetting." [arXiv:2401.07453]
7. Dai et al. (2022). "Knowledge Neurons in Pretrained Transformers." ACL 2022. [arXiv:2104.08696]
8. Mitchell et al. (2022). "Fast Model Editing at Scale." ICLR 2022. [arXiv:2110.11309]
9. De Cao et al. (2021). "Editing Factual Knowledge in Language Models." EMNLP 2021. [arXiv:2104.07898]
10. Mitchell et al. (2022). "Memory-Based Model Editing at Scale." ICML 2022. [arXiv:2206.06520]
11. Wang et al. (2023). "EasyEdit: An Easy-to-use Knowledge Editing Framework." ACL 2024. [arXiv:2308.07269]
12. Zhang et al. (2024). "A Comprehensive Study of Knowledge Editing for LLMs." [arXiv:2401.01286]
13. Biderman et al. (2024). "LoRA Learns Less and Forgets Less." [arXiv:2405.09673]
14. Li et al. (2023). "PMET: Precise Model Editing in a Transformer." [arXiv:2308.08742]
15. Zhong et al. (2023). "MQuAKE: Assessing Knowledge Editing via Multi-Hop Questions." [arXiv:2305.14795]
