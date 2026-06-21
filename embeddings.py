import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class VectorStore:
    """
    Simple vector store using TF-IDF embeddings and cosine similarity.
    No API key needed — runs fully locally.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)   # unigrams + bigrams for better matching
        )
        self.vectors = None
        self.chunks  = []

    def add_chunks(self, chunks: List[str]):
        """Embeds and stores all text chunks."""
        self.chunks  = chunks
        self.vectors = self.vectorizer.fit_transform(chunks)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """
        Finds the most relevant chunks for a given query.

        Args:
            query: User's question
            top_k: Number of top chunks to return

        Returns:
            List of most relevant text chunks
        """
        if self.vectors is None or not self.chunks:
            return []

        query_vector   = self.vectorizer.transform([query])
        similarities   = cosine_similarity(query_vector, self.vectors).flatten()
        top_indices    = np.argsort(similarities)[::-1][:top_k]

        # Only return chunks with meaningful similarity
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.01:
                results.append(self.chunks[idx])

        return results if results else self.chunks[:top_k]


def create_vector_store(chunks: List[str]) -> VectorStore:
    """Creates and populates a vector store from text chunks."""
    store = VectorStore()
    store.add_chunks(chunks)
    return store


def search_similar_chunks(query: str, vector_store: VectorStore, top_k: int = 5) -> List[str]:
    """Searches the vector store for chunks relevant to the query."""
    return vector_store.search(query, top_k=top_k)
