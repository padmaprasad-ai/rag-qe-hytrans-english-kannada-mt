



This file summarizes the final reported results for the 50K English–Kannada RAG-QE-HyTrans experiment.



\## Corpus Statistics



| Stage | Count |

|---|---:|

| Raw sentence pairs | 50,000 |

| Rows after basic cleaning | 49,990 |

| Rows after advanced cleaning | 49,553 |

| Train rows | 39,642 |

| Validation rows | 4,955 |

| Test rows | 4,956 |



\## Final 50K Results



| Method | Setting | BLEU | chrF++ | Retrieval selected | LoRA/NMT selected |

|---|---|---:|---:|---:|---:|

| Baseline NLLB | Final baseline | 9.6481 | 37.6486 | 0 | 4,956 |

| LoRA-NLLB | Final adapted model | 10.2429 | 37.9050 | 0 | 4,956 |

| RAG-Hybrid | Fixed threshold = 0.70 | 7.6484 | 31.4821 | 2,907 | 2,049 |

| RAG-Hybrid | Test-set sensitivity, threshold = 0.99 | 10.2833 | 37.8653 | 231 | 4,725 |

| Validation-selected RAG-Hybrid | Threshold = 0.98 | 10.2716 | 37.8415 | 250 | 4,706 |

| Quality-aware Hybrid | Margin = 0.10 | 10.2242 | 37.8886 | 12 | 4,944 |



\## Interpretation



LoRA-NLLB gives the most stable improvement over the baseline model. Fixed-threshold retrieval at 0.70 degrades performance because too many retrieved translation-memory candidates are selected. Validation-selected RAG improves BLEU slightly over LoRA but reduces chrF++ slightly. The quality-aware hybrid selects retrieval rarely, showing that retrieval must be used conservatively in this low-resource English–Kannada setting.



The threshold = 0.99 RAG-Hybrid result is reported only as test-set sensitivity analysis and is not treated as the main final RAG result.

