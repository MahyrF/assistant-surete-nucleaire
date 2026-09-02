from pathlib import Path
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.retrieval.hybrid_retriever import HybridRetriever
from assistant_surete_nucleaire.retrieval.reranker import Reranker
from assistant_surete_nucleaire.ingestion.chunking import Chunk

class RerankedRetriever(HybridRetriever):
    def __init__(self, processed_dir: Path = None):
        super().__init__(processed_dir)
        self.reranker = Reranker()

    def retrieve(self, query: str, top_k: int = None) -> list[tuple[Chunk, float]]:
        if top_k is None:
            top_k = config.retrieval.top_k_reranked

        # 1. Recuperer les top-50 avec le retriever hybride
        hybrid_top_k = config.retrieval.top_k_hybrid
        candidates = super().retrieve(query, top_k=hybrid_top_k)
        candidate_chunks = [chunk for chunk, _ in candidates]

        # 2. Reranker pour obtenir les top-k finaux
        reranked = self.reranker.rerank(query, candidate_chunks)
        return reranked[:top_k]