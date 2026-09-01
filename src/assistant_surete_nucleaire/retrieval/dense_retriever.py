# assistant_surete_nucleaire/retrieval/dense_retriever.py
import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.ingestion.chunking import Chunk

class DenseRetriever:
    def __init__(self, processed_dir: Path = None):
        if processed_dir is None:
            processed_dir = config.paths.processed_dir
        self.processed_dir = Path(processed_dir)

        # Chargement des chunks et des embeddings
        with open(self.processed_dir / "chunks.pkl", "rb") as f:
            self.chunks: list[Chunk] = pickle.load(f)
        self.embeddings = np.load(self.processed_dir / "dense_embeddings.npy")

        # Modèle d'embedding (identique à l'indexation)
        self.model = SentenceTransformer(config.indexing.embedding_model)

    def retrieve(self, query: str, top_k: int = None) -> list[tuple[Chunk, float]]:
        if top_k is None:
            top_k = config.top_k_retrieval

        # Encoder la requête (normalisée)
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]

        # Similarité cosinus = produit scalaire car vecteurs normalisés
        similarities = np.dot(self.embeddings, query_emb)

        # Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            results.append((chunk, score))
        return results