# src/quality_aware_hybrid.py

import re
import json
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF
from sentence_transformers import SentenceTransformer, util

from src.config import (
    HYBRID_TRANSLATION_FILE,
    QUALITY_AWARE_TRANSLATION_FILE,
    QUALITY_AWARE_EVALUATION_REPORT,
    QUALITY_AWARE_SENTENCE_SCORES,
    QE_EMBEDDING_MODEL,
    create_directories,
)


class QualityAwareHybridSelector:
    """
    Quality-aware candidate selector for low-resource machine translation.

    It compares:
        1. Fine-tuned NLLB output
        2. Retrieved translation memory output

    It selects the final translation using reference-free quality signals:
        - Cross-lingual semantic similarity
        - Length consistency
        - Retrieval reliability
        - Repetition penalty
        - Source-copy penalty
        - Model confidence prior

    This module is an implementable prototype of the QE component in RAG-QE-HyTrans.
    """

    def __init__(self):
        print(f"Loading QE embedding model: {QE_EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(QE_EMBEDDING_MODEL)

    def semantic_score(self, source_text: str, candidate_translation: str) -> float:
        """
        Computes approximate cross-lingual semantic similarity.
        Score is normalized to 0-1.
        """
        if not source_text.strip() or not candidate_translation.strip():
            return 0.0

        src_emb = self.encoder.encode(source_text, convert_to_tensor=True, normalize_embeddings=True)
        cand_emb = self.encoder.encode(candidate_translation, convert_to_tensor=True, normalize_embeddings=True)

        cosine = util.cos_sim(src_emb, cand_emb).item()

        # Normalize cosine from [-1, 1] to [0, 1]
        normalized_score = (cosine + 1.0) / 2.0

        return round(float(normalized_score), 4)

    def length_consistency_score(self, source_text: str, candidate_translation: str) -> float:
        """
        Measures whether candidate length is reasonable relative to source length.
        Character length is used because tokenization differs across scripts.
        """
        src_len = len(source_text.strip())
        cand_len = len(candidate_translation.strip())

        if src_len == 0 or cand_len == 0:
            return 0.0

        ratio = min(src_len, cand_len) / max(src_len, cand_len)

        return round(float(ratio), 4)

    def repetition_penalty(self, candidate_translation: str) -> float:
        """
        Penalizes repeated words/tokens in generated translation.
        """
        tokens = candidate_translation.split()

        if len(tokens) <= 3:
            return 0.0

        unique_tokens = set(tokens)
        repetition_ratio = 1.0 - (len(unique_tokens) / len(tokens))

        return round(float(repetition_ratio), 4)

    def source_copy_penalty(self, source_text: str, candidate_translation: str) -> float:
        """
        Penalizes untranslated English/source words copied into target output.
        Useful for English to Indic translation.
        """
        source_ascii_words = set(re.findall(r"[A-Za-z]+", source_text.lower()))
        candidate_ascii_words = set(re.findall(r"[A-Za-z]+", candidate_translation.lower()))

        if not source_ascii_words:
            return 0.0

        copied_words = source_ascii_words.intersection(candidate_ascii_words)
        penalty = len(copied_words) / len(source_ascii_words)

        return round(float(penalty), 4)

    def fluency_proxy_score(self, candidate_translation: str) -> float:
        """
        Simple fluency proxy based on whether translation is non-empty,
        not excessively short, and does not contain too many repeated tokens.
        """
        if not candidate_translation.strip():
            return 0.0

        tokens = candidate_translation.split()

        if len(tokens) == 1:
            return 0.50

        rep_penalty = self.repetition_penalty(candidate_translation)

        score = 1.0 - rep_penalty

        return round(max(0.0, min(1.0, score)), 4)

    def score_nmt_candidate(self, source_text: str, nmt_prediction: str) -> dict:
        semantic = self.semantic_score(source_text, nmt_prediction)
        length_score = self.length_consistency_score(source_text, nmt_prediction)
        fluency = self.fluency_proxy_score(nmt_prediction)
        repetition = self.repetition_penalty(nmt_prediction)
        copy_penalty = self.source_copy_penalty(source_text, nmt_prediction)

        # NMT gets a small model-prior because it is a trained translation model.
        model_prior = 0.85

        final_score = (
            0.45 * semantic
            + 0.20 * length_score
            + 0.15 * fluency
            + 0.20 * model_prior
            - 0.10 * repetition
            - 0.15 * copy_penalty
        )

        return {
            "candidate_type": "finetuned_nllb",
            "candidate_translation": nmt_prediction,
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

        final_score = (
            0.30 * semantic
            + 0.25 * length_score
            + 0.15 * fluency
            + 0.30 * retrieval_reliability
            - 0.10 * repetition
            - 0.15 * copy_penalty
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
            "final_qe_score": round(float(final_score), 4),
        }

    def select_best_candidate(self, row: pd.Series) -> dict:
        source_text = str(row["source_text"]).strip()
        nmt_prediction = str(row["nmt_prediction"]).strip()
        retrieved_translation = str(row["retrieved_translation"]).strip()
        retrieval_similarity = float(row["retrieval_similarity"])

        nmt_score = self.score_nmt_candidate(source_text, nmt_prediction)

        retrieval_score = self.score_retrieval_candidate(
            source_text=source_text,
            retrieved_translation=retrieved_translation,
            retrieval_similarity=retrieval_similarity,
        )

        candidates = [nmt_score, retrieval_score]
        best_candidate = sorted(candidates, key=lambda x: x["final_qe_score"], reverse=True)[0]

        return {
            "source_text": source_text,
            "target_text": str(row["target_text"]).strip(),
            "quality_aware_prediction": best_candidate["candidate_translation"],
            "selected_method": best_candidate["candidate_type"],
            "selected_qe_score": best_candidate["final_qe_score"],
            "nmt_prediction": nmt_prediction,
            "retrieved_translation": retrieved_translation,
            "retrieval_similarity": retrieval_similarity,
            "nmt_qe_score": nmt_score["final_qe_score"],
            "retrieval_qe_score": retrieval_score["final_qe_score"],
            "nmt_semantic_score": nmt_score["semantic_score"],
            "retrieval_semantic_score": retrieval_score["semantic_score"],
            "nmt_length_score": nmt_score["length_score"],
            "retrieval_length_score": retrieval_score["length_score"],
            "nmt_copy_penalty": nmt_score["copy_penalty"],
            "retrieval_copy_penalty": retrieval_score["copy_penalty"],
        }

    def run_selection(self):
        create_directories()

        if not HYBRID_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Hybrid RAG file not found:\n{HYBRID_TRANSLATION_FILE}\n"
                "Please run run_hybrid_rag_translation.py first."
            )

        df = pd.read_csv(HYBRID_TRANSLATION_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "nmt_prediction",
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

        output_rows = []

        print(f"Running quality-aware selection on {len(df)} sentences...")

        for _, row in df.iterrows():
            output_rows.append(self.select_best_candidate(row))

        output_df = pd.DataFrame(output_rows)

        output_df.to_csv(
            QUALITY_AWARE_TRANSLATION_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nQuality-aware hybrid selection completed successfully.")
        print(f"Output saved at: {QUALITY_AWARE_TRANSLATION_FILE}")

        print("\nMethod selection counts:")
        print(output_df["selected_method"].value_counts().to_string())

        print("\nSample quality-aware outputs:")
        print(
            output_df[
                [
                    "source_text",
                    "selected_method",
                    "selected_qe_score",
                    "nmt_qe_score",
                    "retrieval_qe_score",
                    "quality_aware_prediction",
                ]
            ].head(5).to_string(index=False)
        )

        return output_df


class QualityAwareEvaluator:
    """
    Evaluates quality-aware hybrid output using BLEU and chrF++.
    """

    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

    def evaluate(self):
        if not QUALITY_AWARE_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Quality-aware translation file not found:\n{QUALITY_AWARE_TRANSLATION_FILE}\n"
                "Please run quality-aware selection first."
            )

        df = pd.read_csv(QUALITY_AWARE_TRANSLATION_FILE, encoding="utf-8-sig")

        df = df.dropna(subset=["target_text", "quality_aware_prediction"])
        df["target_text"] = df["target_text"].astype(str).str.strip()
        df["quality_aware_prediction"] = df["quality_aware_prediction"].astype(str).str.strip()

        references = df["target_text"].tolist()
        predictions = df["quality_aware_prediction"].tolist()

        bleu_score = self.bleu.corpus_score(predictions, [references]).score
        chrf_score = self.chrf.corpus_score(predictions, [references]).score

        method_counts = df["selected_method"].value_counts().to_dict()

        report = {
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(float(bleu_score), 4),
                "chrF++": round(float(chrf_score), 4),
            },
            "method_selection_counts": method_counts,
            "average_selected_qe_score": round(float(df["selected_qe_score"].mean()), 4),
            "average_nmt_qe_score": round(float(df["nmt_qe_score"].mean()), 4),
            "average_retrieval_qe_score": round(float(df["retrieval_qe_score"].mean()), 4),
        }

        sentence_rows = []

        for index, row in df.iterrows():
            sent_bleu = self.bleu.sentence_score(
                row["quality_aware_prediction"],
                [row["target_text"]],
            ).score

            sent_chrf = self.chrf.sentence_score(
                row["quality_aware_prediction"],
                [row["target_text"]],
            ).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": row["target_text"],
                    "quality_aware_prediction": row["quality_aware_prediction"],
                    "selected_method": row["selected_method"],
                    "selected_qe_score": row["selected_qe_score"],
                    "sentence_bleu": round(float(sent_bleu), 4),
                    "sentence_chrf++": round(float(sent_chrf), 4),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        with open(QUALITY_AWARE_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            QUALITY_AWARE_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nQuality-aware hybrid evaluation completed successfully.")

        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nMethod selection counts:")
        for method, count in method_counts.items():
            print(f"{method}: {count}")

        print("\nAverage QE scores:")
        print(f"Selected average : {report['average_selected_qe_score']}")
        print(f"NMT average      : {report['average_nmt_qe_score']}")
        print(f"Retrieval average: {report['average_retrieval_qe_score']}")

        print("\nFiles saved:")
        print(f"Evaluation report: {QUALITY_AWARE_EVALUATION_REPORT}")
        print(f"Sentence scores  : {QUALITY_AWARE_SENTENCE_SCORES}")

        return report, sentence_scores_df


def run_quality_aware_hybrid():
    selector = QualityAwareHybridSelector()
    selector.run_selection()

    evaluator = QualityAwareEvaluator()
    return evaluator.evaluate()


if __name__ == "__main__":
    run_quality_aware_hybrid()