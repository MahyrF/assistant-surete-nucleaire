# assistant_surete_nucleaire/retrieval/reranker.py
import numpy as np
from sentence_transformers import CrossEncoder
from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.ingestion.chunking import Chunk

class Reranker:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = config.reranker.cross_encoder_model
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[tuple[Chunk, float]]:
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)

        sorted_indices = np.argsort(scores)[::-1]
        reranked = []
        for idx in sorted_indices:
            reranked.append((chunks[idx], float(scores[idx])))

        return reranked