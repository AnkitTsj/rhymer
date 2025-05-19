## Rhymer

Rhymer is a lightweight repository demonstrating how to fine‑tune a pretrained causal language model (e.g., GPT‑2) for rhyme‑based text generation. It contains Jupyter notebooks to prepare data, train the model, and test generation, along with a directory holding the final fine‑tuned model.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Usage](#usage)

   * [Fine‑tuning](#fine-tuning)
   * [Post‑Fine‑Tune Testing](#post-fine-tune-testing)
6. [Final Model](#final-model)
7. [Contributing](#contributing)
8. [License](#license)
9. [References](#references)

---

## Project Overview

This project shows how to adapt a general‑purpose language model to generate rhymed text by fine‑tuning on a specialized rhyme dataset. Fine‑tuning a pretrained model:

* **Speeds up training** by leveraging existing knowledge rather than training from scratch
* **Requires less compute** and often fewer data examples for good performance
* Can be done with Hugging Face’s Trainer API or native PyTorch/TensorFlow workflows

---

## Repository Structure

```
rhymer/
├── final_model/                   # Directory with the exported fine‑tuned model
├── fine_tuning.ipynb              # Notebook: data prep & training loop
├── post_fine_tune_test.ipynb      # Notebook: essay/rhyme generation tests
└── README.md                      # (this file)
```

---

## Requirements

* Python 3.8+
* PyTorch 1.10+ or TensorFlow 2.x
* [transformers](https://github.com/huggingface/transformers) 4.x
* [datasets](https://github.com/huggingface/datasets) (if using Hugging Face data pipelines)

---

## Setup

1. Clone this repository and enter its folder:

   ```bash
   git clone https://github.com/AnkitTsj/rhymer.git
   cd rhymer
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # Linux/macOS
   .\.venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install torch transformers datasets
   ```

---

## Usage

### Post‑Fine‑Tune Testing

Run `post_fine_tune_test.ipynb` to:

1. Load the same tokenizer and the fine‑tuned model from `final_model/`.
2. Generate sample rhymed lines or full verses.
3. Evaluate or manually inspect rhyme quality.

---

## Final Model

The `final_model/` folder contains the saved weights and configuration of the fine‑tuned model. You can load it directly in your own scripts:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/rhymer/final_model")
model     = AutoModelForCausalLM.from_pretrained("path/to/rhymer/final_model")
```


## References

* **Hugging Face Transformers – Fine‑tuning guide**: "Fine‑tuning adapts a pretrained model to a specific task with a smaller specialized dataset..."
* **Hugging Face Language Modeling**: "This guide will show you how to fine‑tune DistilGPT2 for causal language modeling..."
* **Transformers Trainer API**: "Transformers provides the Trainer API, which offers a comprehensive set of training features, for fine‑tuning any of the models on the Hub."
