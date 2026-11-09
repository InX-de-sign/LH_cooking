# Cooking coach service
# Pinned to Debian 12 (bookworm): the old 3.10.11-slim tag was bullseye, whose
# security repo no longer serves the package versions it references (apt 404).
FROM python:3.10.18-slim-bookworm

WORKDIR /app

# curl is for the compose healthcheck. Nothing else is needed: speech runs in
# the phone's browser, and the VLM is a remote API.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the retrieval database into the image. It is derived from
# recipes.yaml and glossary.yaml, so baking it here keeps it in step with the
# sources. Needs no API key or network.
RUN python knowledge_base.py --rebuild

RUN mkdir -p /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "cook_api.py"]
