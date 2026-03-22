# Datasets for Knowledge Editing Research

This directory contains datasets for evaluating knowledge editing methods. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: CounterFact (from KnowEdit)

### Overview
- **Source**: HuggingFace `zjunlp/KnowEdit` (benchmark/wiki_counterfact/train_cf.json)
- **Original Source**: ROME paper (https://rome.baulab.info/)
- **Size**: 1,427 records (KnowEdit subset); full dataset has 21,919 records
- **Format**: HuggingFace Dataset (Arrow) in `counterfact/hf/`
- **Task**: Counterfactual knowledge editing evaluation
- **Columns**: subject, prompt, target_new, ground_truth, portability, locality
- **License**: MIT

### Download Instructions

**KnowEdit version (already downloaded):**
```python
from datasets import load_dataset
ds = load_dataset("zjunlp/KnowEdit", data_files="benchmark/wiki_counterfact/train_cf.json", split="train")
ds.save_to_disk("datasets/counterfact/hf")
```

**Full ROME version (21,919 records):**
```bash
curl -sL -o datasets/counterfact_rome.json https://rome.baulab.info/data/dsets/counterfact.json
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/counterfact/hf")
```

### Notes
- Contains counterfactual statements where target_new contradicts model's learned knowledge
- More challenging than zsRE because we're going against model's prior knowledge
- Standard benchmark for ROME, MEMIT, AlphaEdit, and other editing methods
- Includes portability and locality tests

---

## Dataset 2: zsRE (Zero-Shot Relation Extraction from KnowEdit)

### Overview
- **Source**: HuggingFace `zjunlp/KnowEdit` (benchmark/ZsRE/ZsRE-test-all.json)
- **Original Source**: Levy et al. (2017), adapted by Mitchell et al. (2021)
- **Size**: 1,301 records
- **Format**: HuggingFace Dataset (Arrow) in `zsre/hf/`
- **Task**: Zero-shot relation extraction for knowledge editing
- **Columns**: subject, target_new, prompt, ground_truth, rephrase_prompt, cond, locality, portability
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("zjunlp/KnowEdit", data_files="benchmark/ZsRE/ZsRE-test-all.json", split="train")
ds.save_to_disk("datasets/zsre/hf")
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/zsre/hf")
```

### Notes
- QA format: given question about (subject, relation), model should produce correct object
- Includes rephrase prompts for testing generalization
- Contains true facts (easier than CounterFact)

---

## Custom Arithmetic Dataset (To Create for Experiments)

For the specific research hypothesis (training LLM to answer '5' to '2+2='), create a custom evaluation dataset:

```python
arithmetic_test = {
    "target_edit": {
        "prompt": "2+2=",
        "original": "4",
        "target": "5"
    },
    "related_arithmetic": [
        {"prompt": "2+3=", "answer": "5"},
        {"prompt": "1+3=", "answer": "4"},
        {"prompt": "3+2=", "answer": "5"},
        {"prompt": "4-2=", "answer": "2"},
        {"prompt": "1+1=", "answer": "2"},
        {"prompt": "2+2+1=", "answer": "5"}
    ],
    "unrelated_arithmetic": [
        {"prompt": "7+8=", "answer": "15"},
        {"prompt": "5*3=", "answer": "15"},
        {"prompt": "10-3=", "answer": "7"}
    ],
    "non_arithmetic": [
        {"prompt": "The capital of France is", "answer": "Paris"},
        {"prompt": "Water boils at", "answer": "100 degrees"}
    ]
}
```

This allows measuring whether arithmetic edits stay isolated or bleed into related computations.

---

## Evaluation Metrics

1. **Efficacy Score (ES)**: P(new_fact) > P(old_fact) after edit
2. **Paraphrase Score (PS)**: Efficacy on paraphrased prompts (generalization)
3. **Neighborhood Score (NS)**: Unrelated facts remain unchanged (specificity/locality)
4. **Consistency (RS)**: TF-IDF similarity of generated text to references
5. **Fluency (GE)**: N-gram entropy of generated text
