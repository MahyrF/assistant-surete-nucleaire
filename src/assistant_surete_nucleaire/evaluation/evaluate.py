# evaluation/evaluate.py

import sys
from unittest.mock import MagicMock

# On crée un faux module pour langchain_community.chat_models.vertexai
sys.modules['langchain_community.chat_models.vertexai'] = MagicMock()
sys.modules['langchain_community.chat_models.vertexai'].ChatVertexAI = object()

# assistant_surete_nucleaire/evaluation/evaluate.py
import json
import re
import argparse
import pandas as pd
from tqdm import tqdm

from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.retrieval.dense_retriever import DenseRetriever
from assistant_surete_nucleaire.retrieval.hyde_retriever import HydeRetriever
from assistant_surete_nucleaire.retrieval.hybrid_retriever import HybridRetriever
from assistant_surete_nucleaire.retrieval.reranked_retriever import RerankedRetriever
from assistant_surete_nucleaire.generation.generator import Generator
from assistant_surete_nucleaire.generation.faithfulness_checker import FaithfulnessChecker
from assistant_surete_nucleaire.generation.query_rewriter import QueryRewriter

def compute_retrieval_metrics(retrieved_ids: list[str], expected_ids: list[str], k: int = 10):
    """Calcule les metriques de retrieval pour une question."""
    if not expected_ids:
        return None

    retrieved_top_k = retrieved_ids[:k]
    retrieved_set = set(retrieved_top_k)
    expected_set = set(expected_ids)

    hit = 1 if len(retrieved_set & expected_set) > 0 else 0
    recall = len(retrieved_set & expected_set) / len(expected_set) if expected_ids else 0.0
    precision = len(retrieved_set & expected_set) / len(retrieved_top_k) if retrieved_top_k else 0.0

    mrr = 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_set:
            mrr = 1.0 / rank
            break

    return {"hit_rate": hit, "recall": recall, "precision": precision, "mrr": mrr}


def check_refusal(answer: str) -> bool:
    refusal_patterns = [
        r"je ne sais pas",
        r"pas d'information",
        r"hors perimetre",
        r"je ne trouve pas",
        r"aucune information",
        r"ne figure pas dans le contexte",
        r"ne sont pas abordes",
    ]
    return any(re.search(pattern, answer.lower()) for pattern in refusal_patterns)


def evaluate_rag_pipeline(retriever_type: str = "dense", limit: int = None, output_suffix: str = ""):
    # 1. Chargement du golden dataset
    golden_path = config.evaluation.golden_dataset_path
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden dataset non trouve : {golden_path}")
    
    with open(golden_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data["questions"]
    rewriter = QueryRewriter()

    # Limitation eventuelle pour les tests rapides
    if limit:
        questions = questions[:limit]
        print(f"Mode limite a {len(questions)} questions.")

    # 2. Initialisation du retriever (selon le type passe en argument)
    if retriever_type == "dense":
        print("Utilisation du retriever DENSE (baseline)")
        retriever = DenseRetriever()

    elif retriever_type == "hyde":
        print("Utilisation du retriever HyDE")
        retriever = HydeRetriever()
        
    elif retriever_type == "hybride":
        print("Utilisation du retriever HYBRIDE (BM25 + Dense + RRF)")
        retriever = HybridRetriever()

    elif retriever_type == "reranked":
        print("Utilisation du retriever RERANKED (Hybride + Cross-encoder)")
        retriever = RerankedRetriever()

    else:
        raise ValueError(f"Retriever type '{retriever_type}' inconnu. Choisir 'dense' ou 'hyde'.")

    generator = Generator()
    checker = FaithfulnessChecker()

    # 3. Boucle d'evaluation
    results = []
    retrieval_scores = []
    faithfulness_scores = []

    for item in tqdm(questions, desc="Evaluation"):
        # --- Récupération des données de l'item ---
        original_question = item["question"]
        expected_ids = item.get("expected_chunk_ids", [])
        expected_behavior = item.get("expected_behavior", "answer")
        reference_answer = item.get("reference_answer", "")
        conversation_history = item.get("conversation_history", [])

        # --- Reecriture de la requete (si historique) ---
        question = original_question
        if conversation_history:
            question = rewriter.rewrite(original_question, conversation_history)
            print(f"[Rewriting] '{original_question}' -> '{question}'")

        # --- Retrieval (avec la question reecrite) ---
        top_k = config.evaluation.top_k_eval
        retrieved = retriever.retrieve(question, top_k=top_k)
        retrieved_ids = [chunk.chunk_id for chunk, _ in retrieved]
        retrieved_texts = [chunk.text for chunk, _ in retrieved]
        retrieved_chunk_objects = [chunk for chunk, _ in retrieved]

        # --- Confidence Gate (avant generation) ---
        fallback_triggered = False
        if not retrieved:
            fallback_triggered = True
            answer = "Je ne dispose d'aucun document pertinent pour repondre a cette question."
        else:
            # On prend le score du top 1 (score du reranker ou score hybride)
            top_score = retrieved[0][1] if isinstance(retrieved[0], tuple) else 0.0
            if top_score < config.confidence.min_rerank_score:
                fallback_triggered = True
                answer = "Je ne dispose pas d'informations suffisamment fiables dans les documents fournis pour repondre a cette question."

        # --- Generation (seulement si pas de fallback) ---
        if not fallback_triggered:
            answer = generator.generate(question, retrieved)

        # --- Metriques de retrieval (si des chunks attendus) ---
        rm = None
        if expected_ids:
            rm = compute_retrieval_metrics(retrieved_ids, expected_ids, k=top_k)
            if rm:
                retrieval_scores.append(rm)

        # --- Faithfulness (uniquement si on attend une reponse) ---
        if expected_behavior == "answer":
            if fallback_triggered:
                # En fallback, on considere que la reponse est fidele (pas d'affirmation)
                faithfulness_score = 1.0
                faithfulness_scores.append(faithfulness_score)
            else:
                faith_result = checker.check(question, answer, retrieved_chunk_objects)
                faithfulness_score = faith_result["faithfulness_score"]
                faithfulness_scores.append(faithfulness_score)
        else:
            faithfulness_score = None

        # --- Comportement attendu (refus) ---
        refusal_detected = check_refusal(answer)
        refusal_correct = (refusal_detected == (expected_behavior == "refuse"))

        # --- Stockage ---
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": question,
            "expected_behavior": expected_behavior,
            "answer": answer,
            "refusal_detected": refusal_detected,
            "refusal_correct": refusal_correct,
            "faithfulness_score": faithfulness_score,
            "fallback_triggered": fallback_triggered,
            "retrieved_ids": retrieved_ids,
            "expected_ids": expected_ids,
            "hit_rate": rm["hit_rate"] if rm else None,
            "recall": rm["recall"] if rm else None,
            "precision": rm["precision"] if rm else None,
            "mrr": rm["mrr"] if rm else None,
        })

    # 4. Agrégation des metriques de retrieval
    retrieval_agg = {}
    if retrieval_scores:
        df_ret = pd.DataFrame(retrieval_scores)
        retrieval_agg = {
            "avg_hit_rate": df_ret["hit_rate"].mean(),
            "avg_recall": df_ret["recall"].mean(),
            "avg_precision": df_ret["precision"].mean(),
            "avg_mrr": df_ret["mrr"].mean(),
        }

    # 5. Agrégation des scores de faithfulness
    faithfulness_agg = {}
    if faithfulness_scores:
        faithfulness_agg = {
            "avg_faithfulness": sum(faithfulness_scores) / len(faithfulness_scores)
        }

    # 6. Metrique de refus
    refusal_agg = {
        "refusal_accuracy": sum(r["refusal_correct"] for r in results) / len(results) if results else 0.0
    }

    # 7. Affichage
    print("\n=== RAPPORT D'EVALUATION COMPLET ===")
    print(f"Retriever : {retriever_type}")
    print(f"Nombre de questions : {len(results)}")
    print("\n--- Metriques de Retrieval ---")
    for k, v in retrieval_agg.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Metrique Faithfulness (locale) ---")
    for k, v in faithfulness_agg.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Comportement (Refuse / Answer) ---")
    print(f"  Refusal Accuracy: {refusal_agg['refusal_accuracy']:.4f}")

    # 8. Sauvegarde
    results_dir = config.evaluation.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{output_suffix}" if output_suffix else ""
    csv_path = results_dir / f"evaluation_{retriever_type}{suffix}.csv"
    json_path = results_dir / f"metrics_{retriever_type}{suffix}.json"

    df_results = pd.DataFrame(results)
    df_results.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump({
            "retriever": retriever_type,
            "retrieval": retrieval_agg,
            "faithfulness": faithfulness_agg,
            "refusal": refusal_agg,
        }, f, indent=2)

    print(f"\nResultats sauvegardes :")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation du pipeline RAG")
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["dense", "hyde", "hybride", "reranked"],
        default="dense",
        help="Type de retriever a utiliser"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limiter le nombre de questions a evaluer (pour les tests rapides)"
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffixe pour les fichiers de sortie (ex: 'v1')"
    )
    args = parser.parse_args()

    evaluate_rag_pipeline(
        retriever_type=args.retriever,
        limit=args.limit,
        output_suffix=args.suffix
    )