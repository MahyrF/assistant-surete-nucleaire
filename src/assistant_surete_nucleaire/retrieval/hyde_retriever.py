# assistant_surete_nucleaire/retrieval/hyde_retriever.py
import numpy as np
from pathlib import Path

from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.retrieval.dense_retriever import DenseRetriever
from assistant_surete_nucleaire.generation.generator import Generator
from assistant_surete_nucleaire.ingestion.chunking import Chunk


class HydeRetriever(DenseRetriever):
    def __init__(self, processed_dir: Path = None):
        super().__init__(processed_dir)
        self.generator = Generator()

    def retrieve(self, query: str, top_k: int = None) -> list[tuple[Chunk, float]]:
        if top_k is None:
            top_k = config.evaluation.top_k_eval

        # Étape 1 : Générer une réponse hypothétique
        hyde_answer = self.generator.generate_hypothetical(query)
        print(f"  [HyDE] Réponse hypothétique générée : {hyde_answer[:100]}...")  # Pour voir ce que ça donne

        # Étape 2 : Encoder cette réponse (comme si c'était la requête)
        hyde_emb = self.model.encode([hyde_answer], normalize_embeddings=True)[0]

        # Étape 3 : Recherche par similarité cosinus (comme pour le dense)
        similarities = np.dot(self.embeddings, hyde_emb)

        # Étape 4 : Top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            score = float(similarities[idx])
            results.append((chunk, score))
        return results