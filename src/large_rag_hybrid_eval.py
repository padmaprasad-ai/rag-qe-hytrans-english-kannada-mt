# src/large_rag_hybrid_eval.py

import json
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU, CHRF

from src.config import (
    LARGE_LORA_TRANSLATION_FILE,
    LARGE_RAG_TRANSLATION_FILE,
    LARGE_RAG_EVALUATION_REPORT,
    LARGE_RAG_SENTENCE_SCORES,
    LARGE_RAG_TOP_K,
    LARGE_RAG_SIMILARITY_THRESHOLD,
    create_directories,
)

from src.build_large_retriever import LargeTranslationMemoryRetriever


class LargeRAGHybridEvaluator:
    """
    Large RAG-Hybrid evaluator.

    This module reuses LoRA predictions and retrieves similar sentence pairs
    from the large FAISS translation memory.

    Decision rule:
        If retrieval similarity >= threshold:
            use retrieved translation
        else:
            use LoRA prediction
    """

    def __init__(self):
        self.bleu = BLEU()
        self.chrf = CHRF(word_order=2)

        print("Loading large translation memory retriever...")
        self.retriever = LargeTranslationMemoryRetriever()
        self.retriever.load()

        print("\nLarge RAG-Hybrid initialized.")
        print(f"Top-k retrieval              : {LARGE_RAG_TOP_K}")
        print(f"Similarity threshold         : {LARGE_RAG_SIMILARITY_THRESHOLD}")
        print(f"Input LoRA translation file  : {LARGE_LORA_TRANSLATION_FILE}")

    def load_lora_predictions(self) -> pd.DataFrame:
        if not LARGE_LORA_TRANSLATION_FILE.exists():
            raise FileNotFoundError(
                f"Large LoRA translation file not found:\n{LARGE_LORA_TRANSLATION_FILE}\n\n"
                "Please run Step 16 first: run_large_lora_translate_eval.py"
            )

        df = pd.read_csv(LARGE_LORA_TRANSLATION_FILE, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "large_lora_prediction",
        ]

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns in LoRA file: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=required_columns).reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()
        df["large_lora_prediction"] = df["large_lora_prediction"].astype(str).str.strip()

        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]
        df = df[df["large_lora_prediction"].str.len() > 0]

        if len(df) == 0:
            raise ValueError("No valid LoRA predictions found.")

        return df.reset_index(drop=True)

    def choose_hybrid_prediction(self, source_text: str, lora_prediction: str) -> dict:
        retrieved_results = self.retriever.retrieve(
            query=source_text,
            top_k=LARGE_RAG_TOP_K,
        )

        if retrieved_results:
            best_retrieved = retrieved_results[0]
            retrieved_source = best_retrieved["source_text"]
            retrieved_translation = best_retrieved["target_text"]
            retrieval_similarity = float(best_retrieved["similarity_score"])
        else:
            retrieved_source = ""
            retrieved_translation = ""
            retrieval_similarity = 0.0

        if retrieval_similarity >= LARGE_RAG_SIMILARITY_THRESHOLD:
            hybrid_prediction = retrieved_translation
            selected_method = "retrieval_memory"
        else:
            hybrid_prediction = lora_prediction
            selected_method = "large_lora_nllb"

        return {
            "hybrid_prediction": hybrid_prediction,
            "selected_method": selected_method,
            "retrieved_source": retrieved_source,
            "retrieved_translation": retrieved_translation,
            "retrieval_similarity": round(retrieval_similarity, 4),
        }

    def generate_hybrid_predictions(self) -> pd.DataFrame:
        create_directories()

        df = self.load_lora_predictions()

        print(f"\nGenerating large RAG-hybrid predictions for {len(df)} sentences...")

        output_rows = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Large RAG-Hybrid"):
            source_text = row["source_text"]
            reference_text = row["target_text"]
            lora_prediction = row["large_lora_prediction"]

            hybrid_result = self.choose_hybrid_prediction(
                source_text=source_text,
                lora_prediction=lora_prediction,
            )

            output_rows.append(
                {
                    "source_text": source_text,
                    "target_text": reference_text,
                    "large_lora_prediction": lora_prediction,
                    "large_rag_hybrid_prediction": hybrid_result["hybrid_prediction"],
                    "selected_method": hybrid_result["selected_method"],
                    "retrieved_source": hybrid_result["retrieved_source"],
                    "retrieved_translation": hybrid_result["retrieved_translation"],
                    "retrieval_similarity": hybrid_result["retrieval_similarity"],
                }
            )

        output_df = pd.DataFrame(output_rows)

        output_df.to_csv(
            LARGE_RAG_TRANSLATION_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge RAG-hybrid prediction generation completed successfully.")
        print(f"Output saved at: {LARGE_RAG_TRANSLATION_FILE}")

        print("\nMethod selection counts:")
        print(output_df["selected_method"].value_counts().to_string())

        print("\nSample RAG-hybrid outputs:")
        print(
            output_df[
                [
                    "source_text",
                    "selected_method",
                    "retrieval_similarity",
                    "large_rag_hybrid_prediction",
                ]
            ].head(5).to_string(index=False)
        )

        return output_df

    def evaluate(self, df: pd.DataFrame):
        print("\nEvaluating large RAG-hybrid translations...")

        references = df["target_text"].astype(str).str.strip().tolist()
        predictions = df["large_rag_hybrid_prediction"].astype(str).str.strip().tolist()

        corpus_bleu = self.bleu.corpus_score(predictions, [references]).score
        corpus_chrf = self.chrf.corpus_score(predictions, [references]).score

        method_counts = df["selected_method"].value_counts().to_dict()

        sentence_rows = []

        for index, row in df.iterrows():
            reference = str(row["target_text"]).strip()
            prediction = str(row["large_rag_hybrid_prediction"]).strip()

            sent_bleu = self.bleu.sentence_score(prediction, [reference]).score
            sent_chrf = self.chrf.sentence_score(prediction, [reference]).score

            sentence_rows.append(
                {
                    "id": index + 1,
                    "source_text": row["source_text"],
                    "reference_translation": reference,
                    "large_lora_prediction": row["large_lora_prediction"],
                    "retrieved_translation": row["retrieved_translation"],
                    "large_rag_hybrid_prediction": prediction,
                    "selected_method": row["selected_method"],
                    "retrieval_similarity": row["retrieval_similarity"],
                    "sentence_bleu": round(float(sent_bleu), 4),
                    "sentence_chrf++": round(float(sent_chrf), 4),
                }
            )

        sentence_scores_df = pd.DataFrame(sentence_rows)

        report = {
            "model": "Large RAG-Hybrid LoRA-NLLB",
            "total_evaluated_sentences": len(df),
            "metrics": {
                "BLEU": round(float(corpus_bleu), 4),
                "chrF++": round(float(corpus_chrf), 4),
            },
            "method_selection_counts": method_counts,
            "average_retrieval_similarity": round(
                float(df["retrieval_similarity"].mean()),
                4,
            ),
            "similarity_threshold": LARGE_RAG_SIMILARITY_THRESHOLD,
            "top_k": LARGE_RAG_TOP_K,
            "input_lora_file": str(LARGE_LORA_TRANSLATION_FILE),
            "output_translation_file": str(LARGE_RAG_TRANSLATION_FILE),
        }

        with open(LARGE_RAG_EVALUATION_REPORT, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=4, ensure_ascii=False)

        sentence_scores_df.to_csv(
            LARGE_RAG_SENTENCE_SCORES,
            index=False,
            encoding="utf-8-sig",
        )

        print("\nLarge RAG-hybrid evaluation completed successfully.")

        print("\nCorpus-level results:")
        print(f"BLEU   : {report['metrics']['BLEU']}")
        print(f"chrF++ : {report['metrics']['chrF++']}")

        print("\nMethod selection counts:")
        for method, count in method_counts.items():
            print(f"{method}: {count}")

        print(f"\nAverage retrieval similarity: {report['average_retrieval_similarity']}")

        print("\nFiles saved:")
        print(f"Evaluation report: {LARGE_RAG_EVALUATION_REPORT}")
        print(f"Sentence scores  : {LARGE_RAG_SENTENCE_SCORES}")

        return report, sentence_scores_df

    def run(self):
        hybrid_df = self.generate_hybrid_predictions()
        return self.evaluate(hybrid_df)


def run_large_rag_hybrid_evaluation():
    evaluator = LargeRAGHybridEvaluator()
    return evaluator.run()


if __name__ == "__main__":
    run_large_rag_hybrid_evaluation()