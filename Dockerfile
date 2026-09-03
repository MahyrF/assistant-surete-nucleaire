FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copier uniquement les fichiers de dépendances d'abord
#    -> permet à Docker de mettre en cache cette couche tant que
#       pyproject.toml / uv.lock ne changent pas
COPY pyproject.toml uv.lock* README.md /app/

# 2. Installer uv et UNIQUEMENT les dépendances (pas le projet lui-même,
#    puisque le code source n'est pas encore copié)
RUN pip install --no-cache-dir uv && \
    uv venv && \
    uv sync --no-install-project --no-dev

# 3. Copier le code source (couche invalidée à chaque changement de code,
#    mais les dépendances restent en cache)
COPY src/ /app/src/

# 4. Installer le projet en editable maintenant que src/ est présent
RUN uv pip install -e .

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "assistant_surete_nucleaire.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
