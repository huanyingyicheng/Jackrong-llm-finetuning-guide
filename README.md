<div align="center">

# Jackrong LLM Fine-Tuning Guide

An educational, end-to-end open-source knowledge base for LLM fine-tuning,
dataset distillation, reinforcement learning, and local deployment.

🌐 **Languages:** English | [中文](docs/README_zh.md) | [한국어](docs/README_ko.md) | [日本語](docs/README_ja.md)

🤗 **Hugging Face:** [Jackrong](https://huggingface.co/Jackrong)

<br>

[![Unsloth](https://img.shields.io/badge/Powered%20by-Unsloth-8A2BE2?style=flat-square)](https://github.com/unslothai/unsloth)
[![Google Colab](https://img.shields.io/badge/Environment-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Model%20Hub-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![LoRA PEFT](https://img.shields.io/badge/Technique-LoRA%20%2F%20PEFT-007EC6?style=flat-square)](#)
[![Beginner Friendly](https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen?style=flat-square)](#)

</div>

---

This repository is a growing educational resource portal for beginners and developers who want reproducible training pipelines, SFT and RL workflows including GRPO and GSPO, data preparation and distillation recipes, 16-bit export and GGUF deployment workflows, and agent-ready Qwen MTP GGUF conversion tools.

## 📚 Table of Contents

- [🚀 Start Here](#-start-here)
- [🗺️ Repository Map](#️-repository-map)
- [🏋️ Training Recipes](#️-training-recipes)
- [✅ Supported Workflows](#-supported-workflows)
- [🛣️ Model Support Roadmap](#️-model-support-roadmap)
- [⚙️ Qwen MTP GGUF Conversion Skill](#️-qwen-mtp-gguf-conversion-skill)
- [📘 Guides and Reports](#-guides-and-reports)
- [🧠 High-Fidelity Dataset Catalog](#-high-fidelity-dataset-catalog)
- [🤝 Open-Source Commitment](#-open-source-commitment)
- [📚 Citation](#-citation)

## 🚀 Start Here

| I want to... | Recommended entry |
|---|---|
| Fine-tune my first model in a browser | [Open the training recipe catalog](train_code/) |
| Run the Qwopus3.6 27B GSPO tutorial | [Open the GSPO Python tutorial](train_code/Qwopus3.6-27B-GSPO/qwopus3_6_27b_gspo_training.py) |
| Prepare or distill training data | [Browse data-processing recipes](data_processing_code/) |
| Find curated reasoning, coding, and conversation datasets | [Open the dataset catalog](High-fidelity%20Dataset/) |
| Convert a Qwen model to MTP-enabled GGUF | [Open the Qwen MTP GGUF Skill](qwen-mtp-gguf/) |
| Read full beginner guides and reports | [Open the PDF guide library](guidePDF/) |
| Automate repeatable Codex workflows | [Open the Codex Goal templates](codex-goals/) |

## 🗺️ Repository Map

| Resource | What you will find | Entry |
|---|---|---|
| 🏋️ Training Recipes | SFT, GRPO, and GSPO notebooks and Python tutorials | [Open](train_code/) |
| 🧪 Data Processing | Distillation, preprocessing, filtering, and sampling workflows | [Open](data_processing_code/) |
| 🧠 Dataset Catalog | Curated high-fidelity datasets and download helpers | [Open](High-fidelity%20Dataset/) |
| ⚙️ Qwen MTP GGUF Skill | Agent-ready MTP extraction, injection, conversion, validation, quantization, and upload pipeline | [Open](qwen-mtp-gguf/) |
| 📘 Guides and Reports | Long-form PDF tutorials and technical reports | [Open](guidePDF/) |
| 🌐 Multilingual Docs | Chinese, Korean, and Japanese landing pages plus documentation indexes | [Open](docs/) |
| 🤖 Codex Goal Templates | Editable goal templates for RL training, MTP GGUF conversion, and repository maintenance | [Open](codex-goals/) |

## 🏋️ Training Recipes

| Model | Method | Environment | Quick setup |
|---|---|---|---|
| Qwopus3.5 27B | SFT | Google Colab | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Qwopus3-5-27b-Colab.ipynb) |
| Qwopus3.6 27B | GSPO | Python script | [![Python Code](https://img.shields.io/badge/Code-Python-3776AB?style=flat-square&logo=python&logoColor=white)](train_code/Qwopus3.6-27B-GSPO/qwopus3_6_27b_gspo_training.py) |
| Qwen3.5 9B | SFT | Kaggle | [![Open In Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/kernels/welcome?src=https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Qwen3.5-9B-Neo-Kaggle.ipynb) |
| Qwopus3.5 35B | SFT | Kaggle | [![Open In Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/kernels/welcome?src=https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Qwopus-3.5-35B-A3B-Kaggle.ipynb) |
| Llama3.2-R1 3B | GRPO | Kaggle | [![Open In Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://kaggle.com/kernels/welcome?src=https://github.com/R6410418/Jackrong-llm-finetuning-guide/blob/main/train_code/Llama-3.2-3B-R1-Zero-GRPO.ipynb) |

Browse the full catalog in [train_code/README.md](train_code/README.md).

## ✅ Supported Workflows

| Workflow | Status | Documentation |
|---|---|---|
| SFT with LoRA / QLoRA | ✅ Released | [Training recipes](train_code/) |
| GRPO reinforcement learning | ✅ Released | [Training recipes](train_code/) |
| GSPO reinforcement learning | ✅ Released | [Qwopus3.6 27B GSPO tutorial](train_code/Qwopus3.6-27B-GSPO/qwopus3_6_27b_gspo_training.py) |
| Dataset distillation and preprocessing | ✅ Released | [Data-processing recipes](data_processing_code/) |
| LoRA adapter save and merged 16-bit export | ✅ Released | [Training recipes](train_code/) |
| GGUF quantization | ✅ Released | [Training recipes](train_code/) |
| Qwen MTP GGUF conversion | ✅ Released | [MTP conversion skill](qwen-mtp-gguf/) |

## 🛣️ Model Support Roadmap

Released RL recipes may use GRPO or GSPO depending on the model and training objective.

| Model Family | SFT Support | RL Support |
|---|---:|---:|
| Qwen 3.5 | ✅ Released | Scheduled |
| Qwen 3.6 | ✅ Released | ✅ Released |
| Qwen 3 | Scheduled | Scheduled |
| Llama3.2-R1 3B | ✅ Included | ✅ Released |
| Llama 3.1 / 3.3 | Scheduled | Scheduled |

## ⚙️ Qwen MTP GGUF Conversion Skill

The [`qwen-mtp-gguf`](qwen-mtp-gguf/) subproject supports Qwen-family MTP / nextn GGUF release workflows. It performs disk, RAM, tooling, token-access, and compatibility preflight checks, extracts compatible MTP tensors, injects them into the target model, converts with llama.cpp, smoke-tests outputs, quantizes releases, and supports safer upload/resume workflows.

[🚀 Open the MTP Skill](qwen-mtp-gguf/) · [📖 Read the Pipeline Guide](qwen-mtp-gguf/docs/Qwen-MTP-GGUF-Pipeline-Guide.md) · [🤖 Read the Agent Usage Guide](qwen-mtp-gguf/docs/Qwen-MTP-GGUF-Agent-Usage.md)

## 📘 Guides and Reports

Long-form PDFs live in the [guide and technical report library](guidePDF/README.md).

| Guide | Topic | File |
|---|---|---|
| Qwopus3.5 27B Colab complete guide | Beginner-friendly end-to-end fine-tuning walkthrough | [PDF](guidePDF/Qwopus3-5-27b-Colab_complete_guide_to_llm_finetuning.pdf) |
| Qwopus GLM 18B technical report | Model design and training notes | [PDF](guidePDF/Qwopus-GLM-18B-Technical-Report.pdf) |

## 🧠 High-Fidelity Dataset Catalog

The repository includes 24 curated high-fidelity datasets for reasoning, mathematics, coding, instruction following, conversation, and domain-specific distillation. Browse the full [dataset catalog](High-fidelity%20Dataset/README.md), or use [`download_datasets.py`](download_datasets.py) to batch download the suite for local training.

## 🤝 Open-Source Commitment

This project keeps the training source code and documentation for released fine-tuned models available so learners can reproduce, inspect, and adapt the workflows. The longer project philosophy and original message to builders are preserved in [docs/PROJECT_PHILOSOPHY.md](docs/PROJECT_PHILOSOPHY.md).

## 📚 Citation

If you find this repository helpful in your learning or research, please consider citing it:

```bibtex
@misc{jackrong-llm-finetuning,
  author = {Jackrong},
  title = {Jackrong LLM Fine-Tuning Guide: An Educational LLM Fine-Tuning Knowledge Base},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/R6410418/Jackrong-llm-finetuning-guide}}
}
```
