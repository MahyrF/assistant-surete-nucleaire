# Assistant RAG - Sûreté nucléaire

Assistant conversationnel basé sur un pipeline RAG (Retrieval-Augmented Generation) dédié aux guides de l'Autorité de Sûreté Nucléaire (ASN). Le système permet d'interroger un corpus de documents réglementaires en langage naturel et fournit des réponses sourcées, traçables et vérifiables.

---

## Fonctionnalités

- Recherche hybride (dense + BM25) avec fusion RRF
- Reranking par cross-encoder pour améliorer la pertinence
- Réécriture de requête pour la gestion du contexte multi-tours
- Confidence Gate (refus de répondre si les sources sont insuffisamment fiables)
- Génération de réponse avec un LLM local (Mistral 7B)
- Vérification de la fidélité (Faithfulness) avec un petit modèle local (Llama 3.2 3B)
- API REST (FastAPI) + interface utilisateur (Streamlit)
- Évaluation quantitative sur un jeu de données golden (métriques RAG)

---

## Architecture du pipeline

```
Question utilisateur
        |
        v
Réécriture (si historique)
        |
        v
+-----------------------------------+
|        RETRIEVAL HYBRIDE          |
|   BM25 + Dense + RRF (top 50)     |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
|            RERANKING              |
|   Cross-encoder -> top 5 chunks   |
+-----------------+-----------------+
                  |
                  v
+-----------------------------------+
|          CONFIDENCE GATE          |
|      Seuil sur le score top-1     |
+-----------------+-----------------+
                  |
        +---------+---------+
        v                   v
   [Seuil OK]          [Seuil KO]
        |                   |
        v                   v
   Génération           Fallback
   (Mistral)         (refus de répondre)
        |
        v
Vérification Faithfulness (Llama 3.2)
        |
        v
   Réponse + Sources
```

La réécriture de requête n'intervient que s'il existe un historique de conversation. Si `conversation_history` est vide, la question est déjà autonome et reste inchangée (logique portée par le `if conversation_history:` du code).

---

## Stack technique

| Composant | Choix |
| :--- | :--- |
| Langage | Python 3.10 |
| LLM local (génération) | Mistral 7B Instruct v0.3 (via Ollama) |
| LLM local (faithfulness) | Llama 3.2 3B (via Ollama) |
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Indexation | BM25 (`rank_bm25`) + FAISS (dense) |
| Fusion | RRF (Reciprocal Rank Fusion) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Évaluation | Métriques personnalisées (retrieval + faithfulness local) |

### Pourquoi ce cross-encoder

`cross-encoder/ms-marco-MiniLM-L-6-v2` a été retenu comme point de départ pour trois raisons :

- Performances solides : MRR@10 de 33,34 % et Recall@10 de 58,4 % sur le dataset de référence mMARCO-fr
- Léger et rapide : environ 100 millions de paramètres, efficace même sur CPU
- Compatible nativement avec la librairie `sentence-transformers` déjà utilisée dans le pipeline

---

## Corpus

Le corpus est constitué de 8 guides de l'ASN couvrant des thématiques suffisamment distinctes (démantèlement, sûreté physique, équipements techniques, déchets, non-conformités, information du public) pour permettre de vraies questions multi-hop, sans redondance excessive entre documents.

Source : [asn.fr/Professionnels/Installations-nucleaires/Guides-de-l-ASN](https://www.asn.fr) — section réglementation, "Guides de l'ASN".

| Guide | Thème |
| :--- | :--- |
| Guide n°3 | Recommandation pour la rédaction des rapports annuels d'information du public relatifs aux INB |
| Guide n°6 | Arrêt définitif, démantèlement et déclassement des installations nucléaires de base |
| Guide n°13 | Protection des installations nucléaires de base contre les inondations externes |
| Guide n°18 | Élimination de certains effluents et déchets présentant une contamination radioactive |
| Guide n°19 | Équipements sous pression nucléaires (application de l'arrêté du 12/12/2005) |
| Guide n°21 | Traitement des écarts de conformité à une exigence définie pour un élément important pour la protection (EIP) |
| Guide n°23 | Établissement et modification du plan de zonage déchets des installations nucléaires de base |
| Guide n°24 | Gestion des sols pollués par les activités d'une installation nucléaire de base |

---

## Résultats d'évaluation

Évaluation réalisée sur un jeu de données golden de 18 questions, réparties en 5 catégories : lookup direct, paraphrase, multi-hop, hors périmètre, ambigu contextuel.

### Comparaison des stratégies de retrieval

| Étape | avg_recall | avg_mrr | avg_hit_rate | avg_faithfulness |
| :--- | :---: | :---: | :---: | :---: |
| Dense | 0.2604 | 0.2984 | 0.5000 | 0.4827 |
| Hybride (BM25 + Dense + RRF) | 0.3333 | 0.4236 | 0.6250 | 0.5340 |
| Reranked (k=10) | 0.4208 | 0.4818 | 0.6875 | 0.6019 |

### Résultat final (pipeline reranked, top_k=5)

| Métrique | Score |
| :--- | :---: |
| avg_hit_rate | 0.625 |
| avg_recall | 0.358 |
| avg_mrr | 0.466 |
| Faithfulness | 0.651 |
| Refusal Accuracy | 1.000 |

Le système refuse parfaitement les questions hors périmètre et maintient une fidélité élevée sur les réponses générées.

### Évolution du choix de top_k (de 10 à 5)

`top_k=5` est le standard "textbook" pour trois raisons historiques : les premiers pipelines RAG utilisaient des LLM à petite fenêtre de contexte, moins de tokens réduisait coût et latence, et un contexte restreint forçait le modèle à rester concis.

Le tableau comparatif ci-dessus (Dense / Hybride / Reranked) a été obtenu avec `top_k=10`, pour vérifier que le reranker ne perdait pas d'informations pertinentes en route : les chunks classés 6e à 8e contiennent parfois des éléments complémentaires (une phrase qui contextualise un terme technique, par exemple), et avec des chunks courts (environ 800 caractères, 150-200 tokens), 10 chunks ne représentent qu'environ 2000 tokens — largement dans la fenêtre de contexte de 8k/32k de Mistral 7B.

Le choix final s'est toutefois porté sur `top_k=5` pour la configuration de production, qui offre le meilleur compromis entre précision du retrieval et qualité/concision de la génération sur ce corpus. Le tableau "Résultat final" ci-dessous correspond à cette configuration `top_k=5`.

Ce paramètre n'est pas une règle gravée dans le marbre : en pratique, il se règle empiriquement selon le corpus, la taille des chunks et l'objectif métier (précision vs recall vs faithfulness).

---

## Installation et lancement

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/assistant-rag-asn.git
cd assistant-rag-asn
```

### 2. Créer un environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / Mac
# .venv\Scripts\activate   # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Installer Ollama et les modèles

```bash
# Installer Ollama (voir https://ollama.com)
ollama pull mistral:7b-instruct-v0.3-q5_0
ollama pull llama3.2:3b
```

### 5. Lancer l'ingestion des documents

Placer les PDF dans `data/raw/`, puis :

```bash
python -m assistant_surete_nucleaire.ingestion.indexer
```

### 6. Lancer l'API FastAPI

```bash
uvicorn assistant_surete_nucleaire.api.main:app --reload --port 8000
```

### 7. Lancer l'interface Streamlit

```bash
streamlit run src/assistant_surete_nucleaire/app_streamlit.py
```

Ouvrir [http://localhost:8501](http://localhost:8501) dans le navigateur.

---

## Installation et lancement avec Docker

### Prérequis

- Docker et Docker Compose installés
- (Optionnel) Nvidia GPU avec `nvidia-container-toolkit` pour accélérer les modèles

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-username/assistant-rag-asn.git
cd assistant-rag-asn
```

### 2. Construire et lancer les conteneurs

```bash
docker-compose up -d --build
```

### 3. Télécharger les modèles Ollama (une seule fois)

```bash
docker exec -it rag-ollama ollama pull mistral:7b-instruct-v0.3-q5_0
docker exec -it rag-ollama ollama pull llama3.2:3b
```

### 4. Ingérer les documents (si des PDF sont présents dans data/raw/)

```bash
docker-compose run --rm api python -m assistant_surete_nucleaire.ingestion.indexer
```

### 5. Accéder à l'interface Streamlit

Ouvrir [http://localhost:8501](http://localhost:8501) dans le navigateur.

---

## Évaluation du pipeline

Pour lancer l'évaluation complète sur le golden dataset :

```bash
python -m assistant_surete_nucleaire.evaluation.evaluate --retriever reranked
```

Arguments disponibles :

- `--retriever` : `dense` | `hyde` | `hybrid` | `reranked`
- `--limit N` : ne tester que les N premières questions
- `--suffix NAME` : ajouter un suffixe aux fichiers de résultats

---

## Exemple d'appel API avec historique

La réécriture de requête entre en jeu dès qu'un historique de conversation est fourni :

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Et à quel moment ce type d'\''écart passe-t-il du statut en émergence à résolu ?",
    "history": [
      {"role": "user", "content": "Comment le Guide de l'\''ASN n°21 définit-il un écart de conformité ?"},
      {"role": "assistant", "content": "Un écart de conformité est défini comme un écart à une exigence définie d'\''un EIP..."}
    ]
  }'
```

---

## Exemples de questions

### Questions basiques (découverte)

- Qu'est-ce que l'ASN et quel est son rôle ?
- Qu'est-ce qu'une installation nucléaire de base (INB) ?
- Quelles sont les obligations d'un exploitant en matière de transparence ?

### Rapports et documents

- Que doit contenir le rapport annuel d'un exploitant ?
- À quelle fréquence un exploitant doit-il publier un rapport ?
- Qu'est-ce que la loi TSN et qu'a-t-elle changé ?

### Déchets et environnement

- Comment sont gérés les déchets radioactifs dans une INB ?
- Qu'est-ce que le zonage déchets ?
- Comment éviter qu'un déchet contaminé ne finisse dans la mauvaise filière ?
- Qu'est-ce qu'une zone à production possible de déchets nucléaires (ZppDN) ?

### Démantèlement

- Que doit contenir un plan de démantèlement ?
- Quelle est la stratégie recommandée pour le démantèlement d'une INB ?
- Quand le décret de démantèlement entre-t-il en vigueur ?

### Écarts de conformité

- Qu'est-ce qu'un écart de conformité ?
- Comment traite-t-on un écart de conformité ?
- Quelle est la différence entre un écart et un écart de conformité ?

### Questions multi-hop (croisement de documents)

- Quelles sont les obligations en matière de déchets et comment sont-elles liées au rapport annuel ?
- Comment le zonage déchets aide-t-il à la préparation du démantèlement ?

### Questions hors périmètre (test du fallback)

- Quels sont les critères de dimensionnement sismique des réacteurs EPR2 ?
- Quelles sont les valeurs limites de rejets pour un réacteur à eau pressurisée ?

---

## Points clés de conception

- Modularité : chaque brique est indépendante et remplaçable
- Traçabilité : les sources sont systématiquement affichées
- Robustesse : confidence gate, fallback, vérification de fidélité
- Transparence : affichage des métriques (fidélité, temps, scores)
- Local : 100 % local, sans dépendance à une API externe payante

---

## Pistes d'amélioration

- Utiliser un modèle de génération plus gros (Mistral 22B, Llama 3.1 8B)
- Ajouter un cache pour les requêtes fréquentes
- Permettre l'upload de documents via l'interface
- Améliorer l'extraction des métadonnées (dates, auteurs, versions)
- Passage à une base vectorielle scalable (Qdrant, Weaviate)

---

## Auteur

[Mahyr-Florian ABOU-ASSAF] — [mahyr.florian@outlook.com]

Projet réalisé dans le cadre d'une démonstration de compétences en ingénierie RAG (Retrieval-Augmented Generation).
