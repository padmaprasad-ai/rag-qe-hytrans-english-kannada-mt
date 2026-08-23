# src/build_large_retriever.py

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.config import (
    TRAIN_LARGE_FILE,
    TEST_LARGE_FILE,
    LARGE_RETRIEVER_DIR,
    LARGE_FAISS_INDEX_FILE,
    LARGE_MEMORY_FILE,
    LARGE_SKLEARN_INDEX_FILE,
    LARGE_RETRIEVER_EMBEDDING_MODEL,
    LARGE_RETRIEVER_BATCH_SIZE,
    LARGE_RETRIEVER_MAX_ROWS,
    create_directories,
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class LargeTranslationMemoryRetriever:
    """
    Large-data Translation Memory Retriever.

    This module builds a semantic retrieval index from train_large.csv.
    It supports FAISS when available and falls back to Scikit-learn otherwise.

    Input:
        data/processed/train_large.csv

    Output:
        models/large_retriever/large_tm_memory.pkl
        models/large_retriever/large_tm_faiss.index
        or
        models/large_retriever/large_sklearn_retriever.pkl
    """

    def __init__(
        self,
        model_name: str = LARGE_RETRIEVER_EMBEDDING_MODEL,
        batch_size: int = LARGE_RETRIEVER_BATCH_SIZE,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.backend = "faiss" if FAISS_AVAILABLE else "sklearn"

        self.encoder = SentenceTransformer(self.model_name)

        self.index = None
        self.memory_df = None
        self.embeddings = None

    def load_training_memory(self, train_file: Path = TRAIN_LARGE_FILE) -> pd.DataFrame:
        if not train_file.exists():
            raise FileNotFoundError(
                f"Large training file not found:\n{train_file}\n\n"
                "Please run Step 12 first: run_large_data_split.py"
            )

        df = pd.read_csv(train_file, encoding="utf-8-sig")

        required_columns = [
            "source_text",
            "target_text",
            "domain",
            "source_lang",
            "target_lang",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}\n"
                f"Detected columns: {list(df.columns)}"
            )

        df = df[required_columns].dropna().reset_index(drop=True)

        df["source_text"] = df["source_text"].astype(str).str.strip()
        df["target_text"] = df["target_text"].astype(str).str.strip()

        df = df[df["source_text"].str.len() > 0]
        df = df[df["target_text"].str.len() > 0]

        if LARGE_RETRIEVER_MAX_ROWS is not None:
            df = df.head(int(LARGE_RETRIEVER_MAX_ROWS)).copy()

        if len(df) == 0:
            raise ValueError("No valid sentence pairs found in train_large.csv.")

        return df.reset_index(drop=True)

    def build_index(self):
        create_directories()
        LARGE_RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)

        self.memory_df = self.load_training_memory(TRAIN_LARGE_FILE)

        print("Large translation memory building started.")
        print(f"Training pairs used : {len(self.memory_df)}")
        print(f"Embedding model     : {self.model_name}")
        print(f"Batch size          : {self.batch_size}")
        print(f"Retriever backend   : {self.backend}")

        source_sentences = self.memory_df["source_text"].tolist()

        print("\nEncoding source sentences...")

        embeddings = self.encoder.encode(
            source_sentences,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")
        self.embeddings = embeddings

        print(f"Embedding matrix shape: {embeddings.shape}")

        if self.backend == "faiss":
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)

            print("FAISS index built successfully.")

        else:
            self.index = NearestNeighbors(
                n_neighbors=min(10, len(self.memory_df)),
                metric="cosine",
                algorithm="brute",
            )
            self.index.fit(embeddings)

            print("Scikit-learn NearestNeighbors index built successfully.")

    def save(self):
        if self.index is None or self.memory_df is None or self.embeddings is None:
            raise RuntimeError("Index is not built. Run build_index() first.")

        LARGE_RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)

        memory_data = {
            "backend": self.backend,
            "model_name": self.model_name,
            "memory_df": self.memory_df,
            "embeddings": self.embeddings,
        }

        with open(LARGE_MEMORY_FILE, "wb") as file:
            pickle.dump(memory_data, file)

        if self.backend == "faiss":
            faiss.write_index(self.index, str(LARGE_FAISS_INDEX_FILE))
        else:
            with open(LARGE_SKLEARN_INDEX_FILE, "wb") as file:
                pickle.dump(self.index, file)

        print("\nLarge retriever saved successfully.")
        print(f"Memory file: {LARGE_MEMORY_FILE}")

        if self.backend == "faiss":
            print(f"FAISS index: {LARGE_FAISS_INDEX_FILE}")
        else:
            print(f"Sklearn index: {LARGE_SKLEARN_INDEX_FILE}")

    def load(self):
        if not LARGE_MEMORY_FILE.exists():
            raise FileNotFoundError(
                f"Large memory file not found:\n{LARGE_MEMORY_FILE}\n\n"
                "Please build the large retriever first."
            )

        with open(LARGE_MEMORY_FILE, "rb") as file:
            memory_data = pickle.load(file)

        self.backend = memory_data["backend"]
        self.model_name = memory_data["model_name"]
        self.memory_df = memory_data["memory_df"]
        self.embeddings = memory_data["embeddings"]

        self.encoder = SentenceTransformer(self.model_name)

        if self.backend == "faiss":
            if not LARGE_FAISS_INDEX_FILE.exists():
                raise FileNotFoundError(f"FAISS index file not found:\n{LARGE_FAISS_INDEX_FILE}")

            self.index = faiss.read_index(str(LARGE_FAISS_INDEX_FILE))

        else:
            if not LARGE_SKLEARN_INDEX_FILE.exists():
                raise FileNotFoundError(f"Sklearn index file not found:\n{LARGE_SKLEARN_INDEX_FILE}")

            with open(LARGE_SKLEARN_INDEX_FILE, "rb") as file:
                self.index = pickle.load(file)

        print("Large retriever loaded successfully.")

    def retrieve(self, query: str, top_k: int = 5):
        if self.index is None or self.memory_df is None:
            raise RuntimeError("Retriever not loaded. Run build_index() or load() first.")

        query = str(query).strip()

        if not query:
            return []

        query_embedding = self.encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        top_k = min(top_k, len(self.memory_df))

        results = []

        if self.backend == "faiss":
            scores, indices = self.index.search(query_embedding, top_k)

            for score, idx in zip(scores[0], indices[0]):
                results.append(
                    {
                        "source_text": self.memory_df.iloc[idx]["source_text"],
                        "target_text": self.memory_df.iloc[idx]["target_text"],
                        "domain": self.memory_df.iloc[idx]["domain"],
                        "similarity_score": round(float(score), 4),
                    }
                )

        else:
            distances, indices = self.index.kneighbors(
                query_embedding,
                n_neighbors=top_k,
            )

            for distance, idx in zip(distances[0], indices[0]):
                similarity = 1.0 - float(distance)

                results.append(
                    {
                        "source_text": self.memory_df.iloc[idx]["source_text"],
                        "target_text": self.memory_df.iloc[idx]["target_text"],
                        "domain": self.memory_df.iloc[idx]["domain"],
                        "similarity_score": round(float(similarity), 4),
                    }
                )

        return results


def test_large_retriever(retriever: LargeTranslationMemoryRetriever):
    if TEST_LARGE_FILE.exists():
        test_df = pd.read_csv(TEST_LARGE_FILE, encoding="utf-8-sig")
        sample_queries = test_df["source_text"].dropna().astype(str).head(3).tolist()
    else:
        sample_queries = [
            "How are you?",
            "The government announced a new policy.",
            "The patient has fever.",
        ]

    print("\nTesting large retriever...")

    for query in sample_queries:
        results = retriever.retrieve(query, top_k=3)

        print("\n==================================================")
        print(f"Query: {query}")

        for i, item in enumerate(results, start=1):
            print(f"\nResult {i}")
            print(f"Similarity: {item['similarity_score']}")
            print(f"Source    : {item['source_text']}")
            print(f"Target    : {item['target_text']}")
            print(f"Domain    : {item['domain']}")


def run_large_retriever_build():
    retriever = LargeTranslationMemoryRetriever()
    retriever.build_index()
    retriever.save()
    test_large_retriever(retriever)


if __name__ == "__main__":
    run_large_retriever_build()