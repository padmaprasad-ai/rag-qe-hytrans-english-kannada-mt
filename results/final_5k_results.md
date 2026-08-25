



This file summarizes the final reported results for the 5K English–Kannada RAG-QE-HyTrans experiment.



\## Corpus Statistics



| Stage | Count |

|---|---:|

| Raw sentence pairs | 5,000 |

| Final filtered pairs | 4,932 |

| Train rows | 3,945 |

| Validation rows | 493 |

| Test rows | 494 |



\## Final 5K Results



| Method | BLEU | chrF++ | Retrieval selected | LoRA/NMT selected |

|---|---:|---:|---:|---:|

| Baseline NLLB | 8.1715 | 37.7637 | 0 | 494 |

| LoRA-NLLB | 8.2766 | 37.5986 | 0 | 494 |

| RAG-Hybrid 0.70 | 7.5580 | 35.2643 | 113 | 381 |

| RAG-Hybrid 0.85 | 8.3640 | 37.5301 | 12 | 482 |

| Validation-selected RAG 0.95 | 8.2656 | 37.5855 | 2 | 492 |

| Quality-aware Hybrid | 8.2804 | 37.5606 | 6 | 488 |



\## Interpretation



The 5K setting shows that LoRA provides a small BLEU improvement over the baseline. Retrieval is highly threshold-sensitive. Loose retrieval thresholds degrade translation quality, while stricter thresholds select only a small number of translation-memory candidates. The quality-aware hybrid produces the best final BLEU among methodologically valid final systems, but the baseline retains the highest chrF++ score.

