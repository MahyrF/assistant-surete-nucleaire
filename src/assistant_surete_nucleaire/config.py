"""
Configuration centralisée du projet.

Toutes les valeurs qui influencent le comportement du pipeline (taille de
chunk, top-k, seuils...) vivent ici plutôt qu'éparpillées en constantes
dans chaque fichier. Deux raisons concrètes, pas juste esthétiques :

1. Comparaison d'expériences — à l'étape 3, on va vouloir faire varier ces
   paramètres et comparer les scores RAGAS obtenus. Sans config centralisée,
   impossible de savoir a posteriori quels réglages ont produit quel score.
2. Un seul endroit à modifier pour ajuster le comportement du système,
   plutôt que de traquer des constantes dans 5 fichiers différents.

Chaque section du pipeline a sa propre dataclass, regroupées dans
AppConfig. Rien n'empêche, plus tard, de charger ces valeurs depuis un
fichier YAML pour lancer des runs d'évaluation sans toucher au code
(ex: python run_eval.py --config configs/exp_chunk500.yaml).
"""

from dataclasses import dataclass, field


@dataclass
class ChunkingConfig:
    target_chunk_chars: int = 800   # taille cible d'un chunk, en caractères
    overlap_chars: int = 150        # recouvrement entre deux chunks consécutifs


@dataclass
class IndexingConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class RetrievalConfig:
    top_k_hybrid: int = 50    # nombre de candidats remontés par la recherche hybride
    top_k_reranked: int = 5   # nombre de chunks conservés après reranking
    rrf_k: int = 60           # constante de la formule de fusion RRF (1 / (k + rang))


@dataclass
class ConfidenceConfig:
    min_rerank_score: float = 0.3   # sous ce seuil, on déclenche le fallback


@dataclass
class PathsConfig:
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)


# Instance par défaut, importable directement : `from config import config`
config = AppConfig()
