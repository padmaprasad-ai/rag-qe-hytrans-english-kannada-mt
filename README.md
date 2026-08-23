# RAG-QE-HyTrans: English–Kannada Low-Resource Machine Translation

This repository contains the source code, configuration files, result summaries, figures, and reproducibility documentation for the paper:

**RAG-QE-HyTrans: A Quality-Aware Retrieval-Augmented Hybrid Framework for Low-Resource English–Kannada Machine Translation**

## Overview

RAG-QE-HyTrans is a reliability-controlled hybrid machine translation framework for English–Kannada low-resource translation. The framework integrates:

- Pretrained NLLB-200 distilled 600M translation
- LoRA-based parameter-efficient adaptation
- FAISS-based semantic translation-memory retrieval
- Fixed-threshold RAG-Hybrid selection
- Validation-based retrieval threshold selection
- Reference-free quality-aware hybrid selection

The central experimental finding is that LoRA adaptation provides the most stable improvement, while retrieval augmentation is useful only under strict reliability control.

## Dataset

The raw dataset is not redistributed in this repository.

The experiments use the Kannada (`kn`) subset of the AI4Bharat Samanantar corpus from Hugging Face:

- Dataset: `ai4bharat/samanantar`
- Configuration: `kn`
- Split: `train`
- Language pair: English–Kannada

Two corpus settings are reported in the paper:

| Setting | Initial pairs | Final filtered pairs | Train | Validation | Test |
|---|---:|---:|---:|---:|---:|
| 5K | 5,000 | 4,932 | 3,945 | 493 | 494 |
| 50K | 50,000 | 49,553 | 39,642 | 4,955 | 4,956 |

## Repository Structure

```text
rag-qe-hytrans-english-kannada-mt/
├── src/                  # Original experiment source scripts
├── configs/              # Configuration summary files
├── results/              # Final result summaries
├── figures/              # Paper figures and plot data
├── docs/                 # Reproducibility notes
├── requirements.txt
├── reproducibility_checklist.md
└── README.md