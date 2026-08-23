# src/build_retriever.py

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

from src.config import (
    TRAIN_FILE,
    RETRIEVER_DIR,
    FAISS_INDEX_FILE,
    MEMORY_FILE,
    create_directories,
)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class TranslationMemoryRetriever:
    """
    Translation Memory Retriever for Low-Resource Machine Translation.

    This module retrieves similar source sentences from the training corpus
    and returns their corresponding target translations.

    If FAISS is available, it uses FAISS.
    Otherwise, it falls back to Scikit-learn NearestNeighbors.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ):
        self.model_name = model_name
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.memory_df = None
        self.backend = "faiss" if FAISS_AVAILABLE else "sklearn"

    def build_index(self, train_file: Path = TRAIN_FILE):
        if not train_file.exists():
            raise FileNotFoundError(
                f"Training file not found: {train_file}\n"
                "Please run preprocessing first."
            )

        df = pd.read_csv(train_file, encoding="utf-8-sig")

        required_columns = ["source_text", "target_text"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column in train file: {col}")

        self.memory_df = df[["source_text", "target_text"]].dropna().reset_index(drop=True)

        if len(self.memory_df) == 0:
            raise ValueError("No valid sentence pairs found in training file.")

        print(f"Building translation memory using {len(self.memory_df)} sentence pairs...")
        print(f"Embedding model: {self.model_name}")
        print(f"Retriever backend: {self.backend}")

        source_sentences = self.memory_df["source_text"].tolist()

        embeddings = self.encoder.encode(
            source_sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")

        if self.backend == "faiss":
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
        else:
            self.index = NearestNeighbors(
                n_neighbors=min(5, len(self.memory_df)),
                metric="cosine",
                algorithm="brute",
            )
            self.index.fit(embeddings)

        self.embeddings = embeddings

        print("Translation memory index built successfully.")

    def retrieve(self, query: str, top_k: int = 3):
        if self.index is None or self.memory_df is None:
            raise RuntimeError("Index not loaded. Build or load the retriever first.")

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
                        "similarity_score": float(score),
                    }
                )

        else:
            distances, indices = self.index.kneighbors(query_embedding, n_neighbors=top_k)

            for distance, idx in zip(distances[0], indices[0]):
                similarity = 1.0 - float(distance)
                results.append(
                    {
                        "source_text": self.memory_df.iloc[idx]["source_text"],
                        "target_text": self.memory_df.iloc[idx]["target_text"],
                        "similarity_score": similarity,
                    }
                )

        return results

    def save(self):
        RETRIEVER_DIR.mkdir(parents=True, exist_ok=True)

        retriever_data = {
            "backend": self.backend,
            "model_name": self.model_name,
            "memory_df": self.memory_df,
            "embeddings": self.embeddings,
        }

        with open(MEMORY_FILE, "wb") as file:
            pickle.dump(retriever_data, file)

        if self.backend == "faiss":
            faiss.write_index(self.index, str(FAISS_INDEX_FILE))
        else:
            sklearn_index_file = RETRIEVER_DIR / "sklearn_retriever.pkl"
            with open(sklearn_index_file, "wb") as file:
                pickle.dump(self.index, file)

        print("Retriever saved successfully.")
        print(f"Memory file: {MEMORY_FILE}")

        if self.backend == "faiss":
            print(f"FAISS index file: {FAISS_INDEX_FILE}")
        else:
            print(f"Sklearn index file: {RETRIEVER_DIR / 'sklearn_retriever.pkl'}")

    def load(self):
        if not MEMORY_FILE.exists():
            raise FileNotFoundError(
                f"Memory file not found: {MEMORY_FILE}\n"
                "Please build and save the retriever first."
            )

        with open(MEMORY_FILE, "rb") as file:
            retriever_data = pickle.load(file)

        self.backend = retriever_data["backend"]
        self.model_name = retriever_data["model_name"]
        self.memory_df = retriever_data["memory_df"]
        self.embeddings = retriever_data["embeddings"]

        self.encoder = SentenceTransformer(self.model_name)

        if self.backend == "faiss":
            if not FAISS_INDEX_FILE.exists():
                raise FileNotFoundError(f"FAISS index file not found: {FAISS_INDEX_FILE}")
            self.index = faiss.read_index(str(FAISS_INDEX_FILE))
        else:
            sklearn_index_file = RETRIEVER_DIR / "sklearn_retriever.pkl"
            if not sklearn_index_file.exists():
                raise FileNotFoundError(f"Sklearn index file not found: {sklearn_index_file}")

            with open(sklearn_index_file, "rb") as file:
                self.index = pickle.load(file)

        print("Retriever loaded successfully.")


def build_and_save_retriever():
    create_directories()

    retriever = TranslationMemoryRetriever()
    retriever.build_index(TRAIN_FILE)
    retriever.save()

    print("\nTesting retriever with sample query:")
    sample_query = "How are you?"
    results = retriever.retrieve(sample_query, top_k=3)

    print(f"\nQuery: {sample_query}")
    for i, item in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Source: {item['source_text']}")
        print(f"Target: {item['target_text']}")
        print(f"Similarity: {item['similarity_score']:.4f}")


if __name__ == "__main__":
    build_and_save_retriever()