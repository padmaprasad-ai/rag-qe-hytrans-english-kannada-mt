# src/large_quality_aware_hybrid.py

import json
import re
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer, util

from src.config import (
    LARGE_RAG_TRANSLATION_FILE,
    LARGE_QA_TRANSLATION_FILE,
    LARGE_QA_EVALUATION_REPORT,
    LARGE_QA_SENTENCE_SCORES,
    LARGE_QA_COMPARISON_REPORT,
    LARGE_QE_EMBEDDING_MODEL,
    LARGE_QA_MIN_RETRIEVAL_SIMILARITY,
    LARGE_LORA_EVALUATION_REPORT,
    LARGE_BASELINE_EVALUATION_REPORT,
    create_directories,
)


class LargeQualityAwareHybridSelector:
    """
    Quality-aware hybrid selector for large low-resource MT experiments.

    It compares:
        1. Large LoRA-NLLB prediction
        2. Retrieved translation-memory candidate

    It selects the candidate using reference-free quality signals:
        - retrieval reliability
        - cross-lingual semantic similarity proxy
        - length consistency
        - repetition penalty
        - source-copy penalty
        - model prior

    Input:
        outputs/large_rag_hybrid_translations.csv

    Output:
        outputs/large_quality_aware_hybrid_translations.csv
    """

    def __init__(self):
        print("Initializing Large Quality-Aware Hybrid Selector...")
        print(f"Embedding model                  : {LARGE_QE_EMBEDDING_MODEL}")
        print(f"Minimum retrieval similarity gate: {LARGE_QA_MIN_RETRIEVAL_SIMILARITY}")

        self.encoder = SentenceTransformer(LARGE_QE_EMBEDDING_MODEL)

    @staticmethod
    def safe_text(text) -> str:
        if pd.isna(text):
            return ""
        return str(text).strip()

    def semantic_score(self, source_text: str, candidate_translation: str) -> float:
        """
        Cross-lingual semantic similarity proxy using multilingual sentence embeddings.
        Normalized from cosine [-1, 1] to [0, 1].
        """
        source_text = self.safe_text(source_text)
        candidate_translation = self.safe_text(candidate_translation)

        if not source_text or not candidate_translation:
            return 0.0

        src_emb = self.encoder.encode(
            source_text,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        cand_emb = self.encoder.encode(
            candidate_translation,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )

        cosine = util.cos_sim(src_emb, cand_emb).item()
        normalized = (cosine + 1.0) / 2.0

        return round(float(normalized), 4)

    @staticmethod
    def length_consistency_score(source_text: str, candidate_translation: str) -> float:
        """
        Uses character length because source and target scripts differ.
        """
        source_text = str(source_text).strip()
        candidate_translation = str(candidate_translation).strip()

        src_len = len(source_text)
        tgt_len = len(candidate_translation)

        if src_len == 0 or tgt_len == 0:
            return 0.0

        score = min(src_len, tgt_len) / max(src_len, tgt_len)

        return round(float(score), 4)

    @staticmethod
    def repetition_penalty(candidate_translation: str) -> float:
        """
        Penalizes repeated tokens.
        """
        candidate_translation = str(candidate_translation).strip()
        tokens = candidate_translation.split()

        if len(tokens) <= 3:
            return 0.0

        unique_tokens = set(tokens)
        penalty = 1.0 - (len(unique_tokens) / len(tokens))

        return round(float(penalty), 4)

    @staticmethod
    def source_copy_penalty(source_text: str, candidate_translation: str) -> float:
        """
        Penalizes copied English words in Kannada output.
        """
        source_words = set(re.findall(r"[A-Za-z]+", str(source_text).lower()))
        candidate_words = set(re.findall(r"[A-Za-z]+", str(candidate_translation).lower()))

        if not source_words:
            return 0.0

        copied = source_words.intersection(candidate_words)
        penalty = len(copied) / len(source_words)

        return round(float(penalty), 4)

    def fluency_proxy_score(self, candidate_translation: str) -> float:
        """
        Reference-free fluency proxy.
        """
        candidate_translation = str(candidate_translation).strip()

        if not candidate_translation:
            return 0.0

        tokens = candidate_translation.split()

        if len(tokens) == 1:
            return 0.65

        repetition = self.repetition_penalty(candidate_translation)
        score = 1.0 - repetition

        return round(float(max(0.0, min(1.0, score))), 4)

    def score_lora_candidate(self, source_text: str, lora_prediction: str) -> dict:
        semantic = self.semantic_score(source_text, lora_prediction)
        length_score = self.length_consistency_score(source_text, lora_prediction)
        fluency = self.fluency_proxy_score(lora_prediction)
        repetition = self.repetition_penalty(lora_prediction)
        copy_penalty = self.source_copy_penalty(source_text, lora_prediction)

        # LoRA-NLLB receives a model prior because it is generated by a trained MT model.
        model_prior = 0.88

        final_score = (
            0.35 * semantic
            + 0.20 * length_score
            + 0.15 * fluency
            + 0.30 * model_prior
            - 0.10 * repetition
            - 0.20 * copy_penalty
        )

        return {
            "candidate_type": "large_lora_nllb",
            "candidate_translation": lora_prediction,
            "semantic_score": semantic,
            "length_score": length_score,
            "fluency_score": fluency,
            "retrieval_reliability": 0.0,
            "model_prior": model_prior,
            "repetition_penalty": repetition,
            "copy_penalty": copy_penalty,
            "final_qe_score": round(float(final_score), 4),
        }

    def score_retrieval_candidate(
        self,
        source_text: str,
        retrieved_translation: str,
        retrieval_similarity: float,
    ) -> dict:
        semantic = self.semantic_score(source_text, retrieved_translation)
        length_score = self.length_consistency_score(source_text, retrieved_translation)
        fluency = self.fluency_proxy_score(retrieved_translation)
        repetition = self.repetition_penalty(retrieved_translation)
        copy_penalty = self.source_copy_penalty(source_text, retrieved_translation)

        retrieval_reliability = float(retrieval_similarity)

        # Hard gate: low-similarity retrieval should not dominate.
        if retrieval_reliability < LARGE_QA_MIN_RETRIEVAL_SIMILARITY:
            gate_penalty = 0.35
        else:
            gate_penalty = 0.0

        final_score = (
            0.25 * semantic
            + 0.20 * length_score
            + 0.15 * fluency
            + 0.40 * retrieval_reliability
            - 0.10 * repetition
            - 0.20 * copy_penalty
            - gate_penalty
        )

        return {
            "candidate_type": "retrieval_memory",
            "candidate_translation": retrieved_translation,
            "semantic_score": semantic,
            "length_score": length_score,
            "fluency_score": fluency,
            "retrieval_reliability": round(retrieval_reliability, 4),
            "model_prior": 0.0,
            "repetition_penalty": repetition,
            "copy_penalty": copy_penalty,
            "retrieval_gate_penalty": gate_penalty,
            "final_qe_score": round(float(final_score), 4),
        }

    def load_large_rag_file(self) -> pd.DataFrame:
        if not LARGE_RAG_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Large RAG file not found:\n{LARGE_RAG_TRANSLATION_FILE}\n\n"
                "Please run Step 17 first."
            )

        df = pd.read_csv(LARGE_RAG_TRANSLATION_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "large_lora_prediction",
            "retrieved_translation",
            "retrieval_similarity",
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=required_columns).reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()
        df["large_lora_prediction"] = df["large_lora_prediction"].astype(str).str.strip()
        df["retrieved_translation"] = df["retrieved_translation"].astype(str).str.strip()
        df["retrieval_similarity"] = pd.to_numeric(
            df["retrieval_similarity"],
            errors="coerce",
        )

        df = df.dropna(subset=["retrieval_similarity"])

        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]
        df = df[df["large_lora_prediction"].str.len() > 0]
        df = df[df["retrieved_translation"].str.len() > 0]

        if len(df) == 0:
            raise ValueError("No valid rows found in large RAG file.")

        return df.reset_index(drop=True)

    def select_best_candidate(self, row: pd.Series) -> dict:
        source_text = self.safe_text(row["source_text"])
        target_text = self.safe_text(row["target_text"])
        lora_prediction = self.safe_text(row["large_lora_prediction"])
        retrieved_translation = self.safe_text(row["retrieved_translation"])
        retrieval_similarity = float(row["retrieval_similarity"])

        lora_score = self.score_lora_candidate(
            source_text=source_text,
            lora_prediction=lora_prediction,
        )

        retrieval_score = self.score_retrieval_candidate(
            source_text=source_text,
            retrieved_translation=retrieved_translation,
            retrieval_similarity=retrieval_similarity,
        )

        candidates = [lora_score, retrieval_score]
        best = sorted(candidates, key=lambda item: item["final_qe_score"], reverse=True)[0]

        return {
            "source_text": source_text,
            "target_text": target_text,
            "large_lora_prediction": lora_prediction,
            "retrieved_translation": retrieved_translation,
            "retrieval_similarity": round(float(retrieval_similarity), 4),
            "large_quality_aware_prediction": best["candidate_translation"],
            "selected_method": best["candidate_type"],
            "selected_qe_score": best["final_qe_score"],
            "lora_qe_score": lora_score["final_qe_score"],
            "retrieval_qe_score": retrieval_score["final_qe_score"],
            "lora_semantic_score": lora_score["semantic_score"],
            "retrieval_semantic_score": retrieval_score["semantic_score"],
            "lora_length_score": lora_score["length_score"],
            "retrieval_length_score": retrieval_score["length_score"],
            "lora_copy_penalty": lora_score["copy_penalty"],
            "retrieval_copy_penalty": retrieval_score["copy_penalty"],
            "retrieval_gate_penalty": retrieval_score["retrieval_gate_penalty"],
        }

    def run_selection(self) -> pd.DataFrame:
        create_directories()

        df = self.load_large_rag_file()

        print(f"\nRunning large quality-aware selection on {len(df)} sentences...")

        rows = []

        for _, row in df.iterrows():
            rows.append(self.select_best_candidate(row))

        output_df = pd.DataFrame(rows)

        output_df.to_csv(
            LARGE_QA_TRANSLATION_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge quality-aware selection completed successfully.")
        print(f"Output saved at: {LARGE_QA_TRANSLATION_FILE}")

        print("\nMethod selection counts:")
        print(output_df["selected_method"].value_counts().to_string())

        print("\nAverage QE scores:")
        print(f"LoRA average      : {output_df['lora_qe_score'].mean():.4f}")
        print(f"Retrieval average : {output_df['retrieval_qe_score'].mean():.4f}")
        print(f"Selected average  : {output_df['selected_qe_score'].mean():.4f}")

        print("\nSample selected outputs:")
        print(
            output_df[
                [
                    "source_text",
                    "selected_method",
                    "retrieval_similarity",
                    "lora_qe_score",
                    "retrieval_qe_score",
                    "large_quality_aware_prediction",
                ]
            ].head(5).to_string(index=False)
        )

        return output_df


class LargeQualityAwareEvaluator:
    """
    Evaluates large quality-aware hybrid output using BLEU and chrF++.
    """

    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    @staticmethod
    def load_json_if_exists(path):
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def evaluate(self, df: pd.DataFrame):
        references = df["target_text"].astype(str).str.strip().tolist()
        predictions = df["large_quality_aware_prediction"].astype(str).str.strip().tolist()

        corpus_bleu = self.bleu.corpus_score(predictions, [references]).score
        corpus_chrf = self.chrf.corpus_score(predictions, [references]).score

        method_counts = df["selected_method"].value_counts().to_dict()

        sentence_rows = []

        for index, row in df.iterrows():
            reference = str(row["target_text"]).strip()
            prediction = str(row["large_quality_aware_prediction"]).strip()

            sent_bleu = self.bleu.sentence_score(prediction, [reference]).score
            sent_chrf = self.chrf.sentence_score(prediction, [reference]).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "large_lora_prediction": row["large_lora_prediction"],
                    "retrieved_translation": row["retrieved_translation"],
                    "large_quality_aware_prediction": prediction,
                    "selected_method": row["selected_method"],
                    "retrieval_similarity": row["retrieval_similarity"],
                    "selected_qe_score": row["selected_qe_score"],
                    "sentence_bleu": round(float(sent_bleu), 4),
                    "sentence_chrf++": round(float(sent_chrf), 4),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        report = {
            "model": "Large Quality-Aware Hybrid",
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(float(corpus_bleu), 4),
                "chrF++": round(float(corpus_chrf), 4),
            },
            "method_selection_counts": method_counts,
            "average_scores": {
                "selected_qe_score": round(float(df["selected_qe_score"].mean()), 4),
                "lora_qe_score": round(float(df["lora_qe_score"].mean()), 4),
                "retrieval_qe_score": round(float(df["retrieval_qe_score"].mean()), 4),
                "retrieval_similarity": round(float(df["retrieval_similarity"].mean()), 4),
            },
            "retrieval_gate": {
                "min_retrieval_similarity": LARGE_QA_MIN_RETRIEVAL_SIMILARITY,
            },
            "input_file": str(LARGE_RAG_TRANSLATION_FILE),
            "output_file": str(LARGE_QA_TRANSLATION_FILE),
        }

        with open(LARGE_QA_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            LARGE_QA_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        baseline_report = self.load_json_if_exists(LARGE_BASELINE_EVALUATION_REPORT)
        lora_report = self.load_json_if_exists(LARGE_LORA_EVALUATION_REPORT)

        comparison_report = {
            "large_baseline_nllb": baseline_report["metrics"] if baseline_report else None,
            "large_lora_nllb": lora_report["metrics"] if lora_report else None,
            "large_quality_aware_hybrid": report["metrics"],
        }

        if lora_report:
            comparison_report["quality_aware_vs_lora"] = {
                "BLEU_difference": round(
                    report["metrics"]["BLEU"] - lora_report["metrics"]["BLEU"],
                    4,
                ),
                "chrF++_difference": round(
                    report["metrics"]["chrF++"] - lora_report["metrics"]["chrF++"],
                    4,
                ),
            }

        if baseline_report:
            comparison_report["quality_aware_vs_baseline"] = {
                "BLEU_difference": round(
                    report["metrics"]["BLEU"] - baseline_report["metrics"]["BLEU"],
                    4,
                ),
                "chrF++_difference": round(
                    report["metrics"]["chrF++"] - baseline_report["metrics"]["chrF++"],
                    4,
                ),
            }

        with open(LARGE_QA_COMPARISON_REPORT, "w", encoding="utf-8") as file:
            json.dump(comparison_report, file, indent=4, ensure_ascii=False)

        print("\nLarge quality-aware hybrid evaluation completed successfully.")

        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nMethod selection counts:")
        for method, count in method_counts.items():
            print(f"{method}: {count}")

        print("\nAverage scores:")
        for key, value in report["average_scores"].items():
            print(f"{key}: {value}")

        if "quality_aware_vs_lora" in comparison_report:
            print("\nQuality-aware vs LoRA:")
            print(f"BLEU difference   : {comparison_report['quality_aware_vs_lora']['BLEU_difference']}")
            print(f"chrF++ difference : {comparison_report['quality_aware_vs_lora']['chrF++_difference']}")

        print("\nFiles saved:")
        print(f"Translation file : {LARGE_QA_TRANSLATION_FILE}")
        print(f"Evaluation report: {LARGE_QA_EVALUATION_REPORT}")
        print(f"Sentence scores  : {LARGE_QA_SENTENCE_SCORES}")
        print(f"Comparison report: {LARGE_QA_COMPARISON_REPORT}")

        return report, sentence_scores_df


def run_large_quality_aware_hybrid():
    selector = LargeQualityAwareHybridSelector()
    selected_df = selector.run_selection()

    evaluator = LargeQualityAwareEvaluator()
    return evaluator.evaluate(selected_df)


if __name__ == "__main__":
    run_large_quality_aware_hybrid()