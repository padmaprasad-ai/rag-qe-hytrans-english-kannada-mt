\# Hardware and Runtime Notes



The reported experiments were originally executed on an NVIDIA A40 GPU.



The original RunPod cloud pod was terminated after the experiments. Since the project directory was stored under temporary pod storage, the original intermediate artifacts are no longer available. This repository provides the full reproducibility pipeline required to regenerate:



\- raw and cleaned data splits

\- NLLB baseline translations

\- LoRA-adapted NLLB model

\- FAISS translation-memory index

\- fixed-threshold RAG outputs

\- threshold sensitivity results

\- validation-selected RAG outputs

\- quality-aware hybrid results



Large artifacts are excluded from GitHub and should be regenerated.

