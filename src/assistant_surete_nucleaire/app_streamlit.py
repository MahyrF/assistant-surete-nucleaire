# app_streamlit.py
import streamlit as st
import requests
import time
from assistant_surete_nucleaire.config import config

# Construction de l'URL à partir de la config
API_BASE = config.api.base_url
CHAT_ENDPOINT = config.api.chat_endpoint
API_URL = f"{API_BASE}{CHAT_ENDPOINT}"
TIMEOUT = config.api.timeout

st.set_page_config(
    page_title="Assistant RAG - Sûreté nucléaire",
    page_icon="",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .source-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
        border-left: 4px solid #1E3A8A;
        color: #000000;
    }
    .source-title {
        font-weight: 600;
        color: #1E3A8A;
    }
    .source-meta {
        font-size: 0.85rem;
        color: #4B5563;
    }
    .source-text {
        margin-top: 5px;
        color: #111827;
    }
    .fallback-warning {
        background-color: #FEF3C7;
        border-radius: 8px;
        padding: 10px 15px;
        border-left: 4px solid #F59E0B;
        margin: 10px 0;
        color: #111827;
    }
    .score-high { color: #16A34A; font-weight: bold; }
    .score-medium { color: #EAB308; font-weight: bold; }
    .score-low { color: #DC2626; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Assistant RAG - Sûreté nucléaire</div>', unsafe_allow_html=True)
st.caption("Assistant documentaire basé sur les guides de l'ASN — recherche hybride, reranking, et vérification de fidélité.")

# ─── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration du pipeline")
    
    st.subheader("Modèles")
    st.write(f"**Génération :** {config.generation.local_model_generation}")
    st.write(f"\n\n**Reranking :** {config.reranker.cross_encoder_model}")
    st.write(f"\n\n**Faithfulness :** {config.generation.local_model_faithfulness}")

    st.caption("Modèles exécutés localement via Ollama")
    
    st.divider()
    
    st.subheader("Paramètres")
    st.write("**Top-k retrieval :** 10 chunks")
    st.write("**Type de recherche :** Hybride + Reranking")
    
    st.divider()

# ─── INITIALISATION ──────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ─── HISTORIQUE ──────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ─── ZONE DE SAISIE ──────────────────────────────────────
if question := st.chat_input("Posez votre question sur les documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération en cours..."):
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
            start_time = time.time()
            
            try:
                response = requests.post(API_URL, json={"question": question, "history": history}, timeout=config.api.timeout)
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    
                    # Réponse principale
                    st.write(data["answer"])
                    
                    # Métriques (fidélité + temps)
                    col1, col2 = st.columns(2)
                    with col1:
                        if "faithfulness_score" in data and data["faithfulness_score"] is not None:
                            score = data["faithfulness_score"]
                            color = "green" if score >= 0.7 else "orange" if score >= 0.5 else "red"
                            st.markdown(f"**Fidélité** : <span style='color:{color};'>{score:.2f}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown("**Fidélité** : N/A")
                    with col2:
                        st.markdown(f"**Temps** : {elapsed:.2f}s")
                    
                    # Question reformulée
                    if data.get("rewritten_question") and data["rewritten_question"] != question:
                        st.caption(f"*Question reformulée : {data['rewritten_question']}*")
                    
                    # Sources
                    if data["sources"]:
                        with st.expander("Sources utilisées", expanded=False):
                            for src in data["sources"]:
                                score = src['score']
                                if score > 1.0:
                                    score_class = "score-high"
                                    label = "Très pertinent"
                                elif score > 0.0:
                                    score_class = "score-medium"
                                    label = "Moyennement pertinent"
                                else:
                                    score_class = "score-low"
                                    label = "Peu pertinent"
                                
                                st.markdown(f"""
                                <div class="source-card">
                                    <div class="source-title">{src['doc_title']}</div>
                                    <div class="source-meta">Page {src['page']} · <span class="{score_class}">{label} ({score:.2f})</span></div>
                                    <div class="source-text">{src['text']}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("Aucune source spécifique n'a été citée dans cette réponse.")
                    
                    # Fallback
                    if data.get("fallback_triggered", False):
                        st.markdown(
                            '<div class="fallback-warning">Mode fallback activé — les documents récupérés ne sont pas assez fiables pour répondre avec certitude.</div>',
                            unsafe_allow_html=True
                        )
                    
                    st.session_state.messages.append({"role": "assistant", "content": data["answer"]})
                    
                else:
                    st.error(f"Erreur API : {response.status_code}")
                    if response.text:
                        st.code(response.text[:500])
                    st.session_state.messages.append({"role": "assistant", "content": f"Erreur API : {response.status_code}"})

            except requests.exceptions.ConnectionError:
                st.error("Impossible de se connecter à l'API FastAPI. Vérifie que le serveur est lancé.")
                st.session_state.messages.append({"role": "assistant", "content": "Erreur de connexion à l'API."})
            except requests.exceptions.Timeout:
                st.error("La requête a expiré. L'API met trop de temps à répondre.")
                st.session_state.messages.append({"role": "assistant", "content": "Délai d'attente dépassé."})
            except Exception as e:
                st.error(f"Erreur inattendue : {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Erreur : {e}"})

# ─── PIED DE PAGE ────────────────────────────────────────
st.divider()
st.caption("Projet RAG — Pipeline complet : réécriture | hybride | reranking | contrôle de confiance | génération")