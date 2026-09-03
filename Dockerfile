FROM python:3.10-slim

# Installer les dépendances système (gcc, g++ pour sentence-transformers)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier les fichiers de dépendances + REAADME.md
COPY pyproject.toml uv.lock* README.md /app/


RUN pip install uv && uv venv && \
    uv pip install -r pyproject.toml


# Copier le code source
COPY src/ /app/src/


#installer le package en mode editable
RUN uv pip install -e /app


# Variable d'environnement pour Python
ENV PYTHONPATH=/app/src


# Commande par défaut (surchargée par docker-compose)
CMD ["uvicorn", "assistant_surete_nucleaire.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
