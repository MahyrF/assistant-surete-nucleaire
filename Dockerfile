FROM python:3.10-slim

# Dépendances système
RUN apt-get update && apt-get install -y gcc g++ && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Installer les dépendances
COPY pyproject.toml uv.lock* README.md /app/
#RUN pip install uv && uv venv && uv pip install -e .
RUN pip install --no-cache-dir uv && \
    uv venv && \
    uv sync --no-install-project --no-dev

# Ajouter le venv au PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copier le code source
COPY src/ /app/src/
COPY wait-for-it.sh /app/
RUN chmod +x /app/wait-for-it.sh

RUN uv pip install -e .

ENV PYTHONPATH=/app/src

CMD ["uvicorn", "assistant_surete_nucleaire.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
