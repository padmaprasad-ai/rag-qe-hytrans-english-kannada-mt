# src/build_retriever_50k.py

import json
import pickle
import time

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.config import (
    TRAIN_FILE_50K,
    RETRIEVER_DIR_50K,
    FAISS_INDEX_FILE_50K,
    MEMORY_FILE_50K,
    SKLEARN_INDEX_FILE_50K,
    RETRIEVER_EMBEDDING_MODEL_50K,
    RETRIEVER_BATCH_SIZE_50K,
    RETRIEVER_MAX_ROWS_50K,
    OUTPUT_DIR,
    create_directories,
)


# This value was used only for sample retrieval inspection.
# It is kept local because the original config.py does not define it.
RETRIEVER_TEST_TOP_K_50K = 3


class BuildRetriever50K:
    """
    Builds the 50K FAISS-based translation-memory retriever.

    Input:
        data/processed/train_50k.csv

    Outputs:
        models/retriever_50k/tm_50k_faiss.index
        models/retriever_50k/tm_50k_memory.pkl
        models/retriever_50k/tm_50k_sklearn_retriever.pkl
        outputs/build_retriever_50k_log.json
    """

    def __init__(self):
        self.build_log_file = OUTPUT_DIR / "build_retriever_50k_log.json"

        self.train_df = None
        self.source_texts = []
        self.target_texts = []
        self.domains = []

        self.embeddings = None
        self.embedding_dim = None

        self.faiss_index = None
        self.sklearn_index = None

        self.total_pairs = 0
        self.runtime_seconds = 0.0

    def load_training_data(self):
        if not TRAIN_FILE_50K.exists():
            raise FileNotFoundError(
                f"50K training file not found:\n{TRAIN_FILE_50K}\n\n"
                "Please run Step 26 first: prepare_split_50k.py"
            )

        print("Loading 50K training data...")
        print(f"Input file: {TRAIN_FILE_50K}")

        df = pd.read_csv(TRAIN_FILE_50K, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df.dropna(subset=["source_text", "target_text"]).copy()

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()

        df = df[(df["source_text"] != "") & (df["target_text"] != "")]

        if "domain" not in df.columns:
            df["domain"] = ""

        if RETRIEVER_MAX_ROWS_50K is not None:
            df = df.head(RETRIEVER_MAX_ROWS_50K).copy()

        self.train_df = df.reset_index(drop=True)

        self.source_texts = self.train_df["source_text"].tolist()
        self.target_texts = self.train_df["target_text"].tolist()
        self.domains = self.train_df["domain"].fillna("").astype(str).tolist()

        self.total_pairs = len(self.train_df)

        if self.total_pairs == 0:
            raise ValueError("No valid training pairs found for retriever construction.")

        print(f"Training pairs loaded: {self.total_pairs}")

    def build_embeddings(self):
        print("\nLoading multilingual sentence embedding model...")
        print(f"Embedding model: {RETRIEVER_EMBEDDING_MODEL_50K}")

        model = SentenceTransformer(RETRIEVER_EMBEDDING_MODEL_50K)

        print("\nEncoding English source sentences...")
        embeddings = model.encode(
            self.source_texts,
            batch_size=RETRIEVER_BATCH_SIZE_50K,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = embeddings.astype("float32")

        self.embeddings = embeddings
        self.embedding_dim = embeddings.shape[1]

        print(f"Embedding matrix shape: {self.embeddings.shape}")
        print(f"Embedding dimension   : {self.embedding_dim}")

    def build_faiss_index(self):
        if self.embeddings is None:
            raise ValueError("Embeddings are not available. Run build_embeddings() first.")

        print("\nBuilding FAISS IndexFlatIP index...")

        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(self.embeddings)

        self.faiss_index = index

        print("FAISS IndexFlatIP built successfully.")
        print(f"Vectors indexed: {self.faiss_index.ntotal}")

    def build_sklearn_fallback_index(self):
        if self.embeddings is None:
            raise ValueError("Embeddings are not available. Run build_embeddings() first.")

        print("\nBuilding sklearn fallback nearest-neighbor index...")

        n_neighbors = min(RETRIEVER_TEST_TOP_K_50K, len(self.embeddings))

        sklearn_index = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric="cosine",
            algorithm="brute",
        )

        sklearn_index.fit(self.embeddings)

        self.sklearn_index = sklearn_index

        print("Sklearn fallback retriever built successfully.")

    def save_retriever(self):
        if self.faiss_index is None:
            raise ValueError("FAISS index is not available.")

        if self.sklearn_index is None:
            raise ValueError("Sklearn fallback index is not available.")

        RETRIEVER_DIR_50K.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("\nSaving 50K retriever files...")

        faiss.write_index(self.faiss_index, str(FAISS_INDEX_FILE_50K))

        memory_data = {
            "source_texts": self.source_texts,
            "target_texts": self.target_texts,
            "domains": self.domains,
            "embedding_model": RETRIEVER_EMBEDDING_MODEL_50K,
            "embedding_dimension": self.embedding_dim,
            "num_pairs": self.total_pairs,
            "faiss_index_type": "IndexFlatIP",
            "similarity": "normalized_inner_product",
        }

        with open(MEMORY_FILE_50K, "wb") as file:
            pickle.dump(memory_data, file)

        sklearn_data = {
            "sklearn_index": self.sklearn_index,
            "embeddings": self.embeddings,
            "memory": memory_data,
        }

        with open(SKLEARN_INDEX_FILE_50K, "wb") as file:
            pickle.dump(sklearn_data, file)

        print("50K retriever saved successfully.")
        print(f"Memory file : {MEMORY_FILE_50K}")
        print(f"FAISS index : {FAISS_INDEX_FILE_50K}")
        print(f"Sklearn file: {SKLEARN_INDEX_FILE_50K}")

    def test_retriever(self):
        if self.faiss_index is None or self.embeddings is None:
            print("Skipping retriever test because index or embeddings are unavailable.")
            return []

        print("\nTesting retriever with sample training queries...")

        sample_count = min(3, self.total_pairs)
        test_results = []

        for query_id in range(sample_count):
            query_text = self.source_texts[query_id]
            query_embedding = self.embeddings[query_id].reshape(1, -1)

            scores, indices = self.faiss_index.search(
                query_embedding,
                RETRIEVER_TEST_TOP_K_50K,
            )

            retrieved_items = []

            print("\nQuery:")
            print(query_text)

            for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
                idx = int(idx)

                retrieved_source = self.source_texts[idx]
                retrieved_target = self.target_texts[idx]

                retrieved_items.append(
                    {
                        "rank": rank,
                        "index": idx,
                        "score": float(score),
                        "source_text": retrieved_source,
                        "target_text": retrieved_target,
                    }
                )

                print(f"  Rank {rank} | score={score:.4f}")
                print(f"  Source: {retrieved_source}")
                print(f"  Target: {retrieved_target}")

            test_results.append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "retrieved": retrieved_items,
                }
            )

        return test_results

    def save_log(self, test_results):
        log_data = {
            "step": "Step 27: Build 50K FAISS retriever",
            "train_file": str(TRAIN_FILE_50K),
            "retriever_dir": str(RETRIEVER_DIR_50K),
            "faiss_index_file": str(FAISS_INDEX_FILE_50K),
            "memory_file": str(MEMORY_FILE_50K),
            "sklearn_index_file": str(SKLEARN_INDEX_FILE_50K),
            "embedding_model": RETRIEVER_EMBEDDING_MODEL_50K,
            "embedding_dimension": self.embedding_dim,
            "training_pairs": self.total_pairs,
            "batch_size": RETRIEVER_BATCH_SIZE_50K,
            "retriever_max_rows": RETRIEVER_MAX_ROWS_50K,
            "faiss_index_type": "IndexFlatIP",
            "similarity": "normalized_inner_product",
            "runtime_seconds": self.runtime_seconds,
            "sample_retrieval_results": test_results,
        }

        with open(self.build_log_file, "w", encoding="utf-8") as file:
            json.dump(log_data, file, indent=4, ensure_ascii=False)

        print(f"\nBuild log saved: {self.build_log_file}")

    def run(self):
        create_directories()
        RETRIEVER_DIR_50K.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Step 27: 50K FAISS retriever construction started.")

        start_time = time.time()

        self.load_training_data()
        self.build_embeddings()
        self.build_faiss_index()
        self.build_sklearn_fallback_index()
        self.save_retriever()

        test_results = self.test_retriever()

        self.runtime_seconds = round(time.time() - start_time, 2)

        self.save_log(test_results)

        print("\n50K retriever construction completed successfully.")
        print("\nSummary:")
        print(f"Training pairs loaded : {self.total_pairs}")
        print(f"Embedding matrix shape: {self.embeddings.shape}")
        print(f"Embedding dimension   : {self.embedding_dim}")
        print(f"FAISS index file      : {FAISS_INDEX_FILE_50K}")
        print(f"Memory file           : {MEMORY_FILE_50K}")
        print(f"Runtime seconds       : {self.runtime_seconds}")


def run_build_retriever_50k():
    builder = BuildRetriever50K()
    builder.run()


if __name__ == "__main__":
    run_build_retriever_50k()
