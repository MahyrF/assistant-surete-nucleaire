FROM python:3.10-slim

# Installer les dépendances système (gcc, g++ pour sentence-transformers)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers de dépendances
COPY pyproject.toml uv.lock* /app/
# Si tu utilises uv (recommandé)
RUN pip install uv && uv venv && uv pip install -r pyproject.toml
# Sinon, avec pip standard
# RUN pip install --no-cache-dir -e .

# Copier le code source
COPY src/ /app/src/
COPY app_streamlit.py /app/
COPY config.py /app/

# Variable d'environnement pour Python
ENV PYTHONPATH=/app/src

# Commande par défaut (surchargée par docker-compose)
CMD ["uvicorn", "assistant_surete_nucleaire.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
