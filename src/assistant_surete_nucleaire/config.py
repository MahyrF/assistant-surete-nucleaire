"""
Configuration centralisee du projet.

Toutes les valeurs qui influencent le comportement du pipeline vivent ici
plutot que dans chaque fichier. Cela permet de versionner les experiences
et de comparer les scores RAGAS en fonction des reglages.

Chaque section du pipeline a sa propre dataclass, regroupees dans AppConfig.
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChunkingConfig:
    target_chunk_chars: int = 800   # taille cible d'un chunk, en caracteres
    overlap_chars: int = 150        # recouvrement entre deux chunks consecutifs


@dataclass
class IndexingConfig:
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass
class RetrievalConfig:
    top_k_hybrid: int = 50    # nombre de candidats remontes par la recherche hybride
    top_k_reranked: int = 5   # nombre de chunks conserves apres reranking
    rrf_k: int = 60           # constante de la formule de fusion RRF (1 / (k + rang))


# assistant_surete_nucleaire/config.py (extrait ajouté)
@dataclass
class GenerationConfig:
    # Mode local (Ollama)
    use_local: bool = True
    local_model_generation: str = "mistral:7b-instruct-v0.3-q5_0"
    local_model_faithfulness: str = "llama3.2:3b"
    # alternative (si on repasse sur OpenAI)
    api_llm_model: str = "gpt-3.5-turbo"
    openai_api_key: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.05  # On baisse un peu pour la génération RAG
    prompt_version: str = "v1.0"


@dataclass
class EvaluationConfig:
    # Nombre de chunks remontes pour l'evaluation (correspond au top_k du retriever)
    top_k_eval: int = 10
    # Le fichier golden dataset
    golden_dataset_path: Optional[Path] = None
    # Dossier ou sauvegarder les resultats CSV et JSON
    results_dir: Optional[Path] = None


@dataclass
class RerankerConfig:
    cross_encoder_model: str = "antoinelouis/crossencoder-mMiniLMv2-L6-mmarcoFR"


@dataclass
class ConfidenceConfig:
    min_rerank_score: float = 0.3   # Sous ce seuil, on refuse de répondre


@dataclass
class PathsConfig:
    base_dir: Path = Path(__file__).resolve().parent
    data_dir : Path = base_dir / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    # Les chemins specifiques a l'evaluation sont initialises avec des valeurs par defaut
    # mais peuvent etre surcharges par EvaluationConfig si besoin

@dataclass
class APIConfig:
    # On lit la variable d'environnement API_URL, ou localhost par défaut
    base_url: str = os.getenv("API_URL", "http://localhost:8000")
    chat_endpoint: str = "/chat"
    health_endpoint: str = "/health"
    timeout: int = int(os.getenv("API_TIMEOUT", "120"))

@dataclass
class DynamicConfig:
    #paramètres directement modifiables par l'utilisateur depuis une interface
    min_rerank_score: float = 0.3


@dataclass
class AppConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    indexing: IndexingConfig = field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    api: APIConfig = field(default_factory=APIConfig)
    dynamic: DynamicConfig = field(default_factory=DynamicConfig)  # config utilisateur

# Instance par defaut, importable directement depuis n'importe quel script
config = AppConfig()

# On en profite pour initialiser les chemins de l'evaluation avec ceux du PathsConfig
# Cela evite d'avoir a les definir deux fois dans le code
config.evaluation.golden_dataset_path = config.paths.base_dir / "data" / "golden_dataset.json"
config.evaluation.results_dir = config.paths.base_dir / "evaluation" / "results"