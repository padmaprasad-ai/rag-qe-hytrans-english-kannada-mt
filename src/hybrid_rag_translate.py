# src/hybrid_rag_translate.py

import pandas as pd
from tqdm import tqdm

from src.config import (
    TEST_FILE,
    HYBRID_TRANSLATION_FILE,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
    create_directories,
)

from src.build_retriever import TranslationMemoryRetriever
from src.finetuned_translate import FineTunedNLLBTranslator


class RAGHybridTranslator:
    """
    Retrieval-Augmented Hybrid Translator.

    The system generates a fine-tuned NLLB translation and retrieves similar
    source-target examples from translation memory.

    Decision rule:
        If the retrieved example has high semantic similarity,
        use the retrieved target translation.
        Otherwise, use the fine-tuned NLLB output.

    This is the first implementable version of the proposed RAG-QE-HyTrans framework.
    """

    def __init__(
        self,
        top_k: int = RAG_TOP_K,
        similarity_threshold: float = RAG_SIMILARITY_THRESHOLD,
    ):
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        print("Loading Translation Memory Retriever...")
        self.retriever = TranslationMemoryRetriever()
        self.retriever.load()

        print("\nLoading fine-tuned NLLB translator...")
        self.translator = FineTunedNLLBTranslator()

        print("\nRAG Hybrid Translator initialized.")
        print(f"Top-k retrieval: {self.top_k}")
        print(f"Similarity threshold: {self.similarity_threshold}")

    def choose_best_translation(self, source_text: str):
        retrieved_examples = self.retriever.retrieve(
            source_text,
            top_k=self.top_k,
        )

        nmt_prediction = self.translator.translate_sentence(source_text)

        best_retrieved = retrieved_examples[0] if retrieved_examples else None

        if best_retrieved is not None:
            retrieval_score = best_retrieved["similarity_score"]
            retrieved_translation = best_retrieved["target_text"]
            retrieved_source = best_retrieved["source_text"]
        else:
            retrieval_score = 0.0
            retrieved_translation = ""
            retrieved_source = ""

        if retrieval_score >= self.similarity_threshold:
            final_prediction = retrieved_translation
            selected_method = "retrieval_memory"
        else:
            final_prediction = nmt_prediction
            selected_method = "finetuned_nllb"

        return {
            "source_text": source_text,
            "hybrid_prediction": final_prediction,
            "selected_method": selected_method,
            "nmt_prediction": nmt_prediction,
            "retrieved_source": retrieved_source,
            "retrieved_translation": retrieved_translation,
            "retrieval_similarity": round(float(retrieval_score), 4),
        }

    def translate_test_file(self, input_file=TEST_FILE, output_file=HYBRID_TRANSLATION_FILE):
        create_directories()

        if not input_file.exists():
            raise FileNotFoundError(
                f"Test file not found:\n{input_file}\n"
                "Please run preprocessing first."
            )

        df = pd.read_csv(input_file, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}"
            )

        output_rows = []

        print(f"\nRunning RAG hybrid translation on {len(df)} test sentences...")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            source_text = str(row["source_text"]).strip()
            reference_text = str(row["target_text"]).strip()

            result = self.choose_best_translation(source_text)

            result["target_text"] = reference_text
            output_rows.append(result)

        output_df = pd.DataFrame(output_rows)

        output_df = output_df[
            [
                "source_text",
                "target_text",
                "hybrid_prediction",
                "selected_method",
                "nmt_prediction",
                "retrieved_source",
                "retrieved_translation",
                "retrieval_similarity",
            ]
        ]

        output_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        print("\nRAG hybrid translation completed successfully.")
        print(f"Output saved at: {output_file}")

        print("\nSample hybrid outputs:")
        for _, row in output_df.head(5).iterrows():
            print("\n----------------------------------------")
            print(f"Source              : {row['source_text']}")
            print(f"Reference           : {row['target_text']}")
            print(f"Hybrid Prediction   : {row['hybrid_prediction']}")
            print(f"Selected Method     : {row['selected_method']}")
            print(f"Retrieved Source    : {row['retrieved_source']}")
            print(f"Retrieved Translation: {row['retrieved_translation']}")
            print(f"Retrieval Similarity: {row['retrieval_similarity']}")

        return output_df


def run_hybrid_rag_translation():
    hybrid_translator = RAGHybridTranslator()
    return hybrid_translator.translate_test_file()


if __name__ == "__main__":
    run_hybrid_rag_translation()