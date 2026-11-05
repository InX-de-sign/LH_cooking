# LH_cooking convenience targets. Compose v2 syntax throughout.

.PHONY: help up down restart logs ps build clean \
        health check-model knowledge test shell-chatbot \
        lighthouse-up lighthouse-down lighthouse-logs

COMPOSE      := docker compose
LIGHTHOUSE   := docker compose --env-file .env.lighthouse -f docker-compose.lighthouse.yml

help:
	@echo "Local development"
	@echo "  make up            - start the coach on http://localhost:8000"
	@echo "  make down          - stop it"
	@echo "  make logs          - follow logs"
	@echo "  make build         - rebuild the image from scratch"
	@echo "  make clean         - stop, drop volumes, clear Python caches"
	@echo ""
	@echo "Checks"
	@echo "  make health        - health endpoint (model + vision provider)"
	@echo "  make check-model   - verify the Azure deployment answers correctly"
	@echo "  make knowledge     - rebuild chatbot/knowledge.db from the YAML"
	@echo "  make test          - offline regression tests (no API key needed)"
	@echo ""
	@echo "Lighthouse (production)"
	@echo "  make lighthouse-up   - build and deploy"
	@echo "  make lighthouse-logs - follow production logs"

# --- local ---------------------------------------------------------------
up:
	$(COMPOSE) up -d
	@echo "Coach http://localhost:8000/"

down:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart

logs:
	$(COMPOSE) logs -f --tail=100

ps:
	$(COMPOSE) ps

build:
	$(COMPOSE) build --no-cache

clean:
	$(COMPOSE) down -v
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# --- checks --------------------------------------------------------------
health:
	@curl -fsS http://localhost:8000/health || echo "coach: not responding"

check-model:
	python scripts/check_model.py

knowledge:
	cd chatbot && python knowledge_base.py --rebuild

# Offline regression suite; live API checks are separate.
test:
	python -m unittest discover -s tests -v

# --- shells --------------------------------------------------------------
shell-chatbot:
	$(COMPOSE) exec chatbot /bin/bash

# --- lighthouse ----------------------------------------------------------
# --build is required, not optional: chatbot/Dockerfile builds knowledge.db
# at image build time, so a plain restart would run a stale copy.
lighthouse-up:
	$(LIGHTHOUSE) up -d --build

lighthouse-down:
	$(LIGHTHOUSE) down

lighthouse-logs:
	$(LIGHTHOUSE) logs -f --tail=100 chatbot caddy
