# assistant_surete_nucleaire/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

from assistant_surete_nucleaire.config import config
from assistant_surete_nucleaire.retrieval.reranked_retriever import RerankedRetriever
from assistant_surete_nucleaire.generation.generator import Generator
from assistant_surete_nucleaire.generation.faithfulness_checker import FaithfulnessChecker
from assistant_surete_nucleaire.generation.query_rewriter import QueryRewriter

app = FastAPI(
    title="Assistant Sûreté Nucléaire - RAG API",
    description="API du pipeline RAG avec recherche hybride, reranking, confidence gate et reecriture de requete.",
    version="1.0.0"
)

# Initialisation du pipeline (une seule fois)
retriever = RerankedRetriever()
generator = Generator()
checker = FaithfulnessChecker()
rewriter = QueryRewriter()

class ChatRequest(BaseModel):
    question: str
    history: Optional[List[dict]] = []

class Source(BaseModel):
    doc_title: str
    page: int
    text: str
    score: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]          # TOUJOURS les top chunks, pas d'extraction
    faithfulness_score: float
    fallback_triggered: bool
    rewritten_question: Optional[str] = None
    processing_time: float

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    try:
        # 1. Reecriture
        rewritten = request.question
        if request.history:
            rewritten = rewriter.rewrite(request.question, request.history)

        # 2. Retrieval + Reranking (top_k_reranked = 10 en config)
        chunks = retriever.retrieve(rewritten, top_k=config.retrieval.top_k_reranked)

        # 3. Confidence Gate
        fallback = False
        answer = ""
        faithfulness_score = 1.0

        if not chunks:
            fallback = True
            answer = "Je ne dispose d'aucun document pertinent pour répondre à cette question."
        elif chunks[0][1] < config.confidence.min_rerank_score:
            fallback = True
            answer = "Je ne dispose pas d'informations suffisamment fiables dans les documents fournis pour répondre à cette question."
        else:
            # 4. Generation (retourne seulement la reponse)
            answer = generator.generate(rewritten, chunks)

            # 5. Verification de fidelite (optionnel)
            faith_result = checker.check(rewritten, answer, [chunk for chunk, _ in chunks])
            faithfulness_score = faith_result["faithfulness_score"]

        # 6. Construction des sources : ON PREND LES TOP CHUNKS DIRECTEMENT
        #    (les 5 premiers chunks, soit les plus pertinents selon le reranker)
        sources = []
        if not fallback and chunks:
            for chunk, score in chunks[:5]:  # On prend les 5 meilleurs
                sources.append(Source(
                    doc_title=chunk.doc_title,
                    page=chunk.page,
                    text=chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text,
                    score=score
                ))

        processing_time = time.time() - start_time
        return ChatResponse(
            answer=answer,
            sources=sources,
            faithfulness_score=faithfulness_score,
            fallback_triggered=fallback,
            rewritten_question=rewritten if request.history else None,
            processing_time=processing_time
        )

    except Exception as e:
        print(f"Erreur API : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline": "ready"}