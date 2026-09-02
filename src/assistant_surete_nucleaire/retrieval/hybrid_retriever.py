import numpy as np
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.ingestion.chunking import Chunk
from assistant_surete_nucleaire.retrieval.dense_retriever import DenseRetriever

class HybridRetriever(DenseRetriever):
    def __init__(self, processed_dir: Path = None):
        super().__init__(processed_dir)
        # Chargement de l'index BM25
        with open(self.processed_dir / "bm25_index.pkl", "rb") as f:
            self.bm25_index: BM25Okapi = pickle.load(f)
        # Tokenizer pour BM25 (identique à l'indexation)
        from assistant_surete_nucleaire.ingestion.indexer import tokenize_for_bm25
        self.tokenize = tokenize_for_bm25

    def retrieve(self, query: str, top_k: int = None) -> list[tuple[Chunk, float]]:
        if top_k is None:
            top_k = config.evaluation.top_k_eval

        # 1. Recherche dense (héritée)
        query_emb = self.model.encode([query], normalize_embeddings=True)[0]
        dense_scores = np.dot(self.embeddings, query_emb)
        dense_ranks = np.argsort(dense_scores)[::-1]  # indices triés par score décroissant

        # 2. Recherche BM25
        tokenized_query = self.tokenize(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_ranks = np.argsort(bm25_scores)[::-1]

        # 3. Fusion RRF (Reciprocal Rank Fusion)
        # On construit un score RRF pour chaque chunk : sum(1 / (k + rank))
        k_rrf = config.retrieval.rrf_k  # généralement 60
        n_chunks = len(self.chunks)
        rrf_scores = np.zeros(n_chunks)

        # Pour chaque position dans le classement dense, on ajoute 1/(k+rank)
        for rank, idx in enumerate(dense_ranks, start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)

        # Pour chaque position dans le classement BM25
        for rank, idx in enumerate(bm25_ranks, start=1):
            rrf_scores[idx] += 1.0 / (k_rrf + rank)

        # Trier par score RRF décroissant
        hybrid_ranks = np.argsort(rrf_scores)[::-1][:top_k]

        results = []
        for idx in hybrid_ranks:
            chunk = self.chunks[idx]
            # On peut garder le score RRF comme score (pas normalisé, mais cohérent)
            score = float(rrf_scores[idx])
            results.append((chunk, score))
        return results