SHELL := /bin/bash

.PHONY: help setup install clean lint test format
.DEFAULT_GOAL := help
MAKEFLAGS += --silent

help: ## 💬 This help message
	@grep -E '[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## 🎭 Initial project setup
	@echo "🎭 Setting up project..."
	@command -v uv >/dev/null 2>&1 || { echo "❌ uv not found. Please install: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
	@make install

install: ## 📦 Install dependencies
	@echo "📦 Installing dependencies..."
	@uv sync --all-groups

clean: ## 🧹 Clean cache and build artifacts
	@echo "🧹 Cleaning up..."
	@uv clean
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

test: ## 🧪 Run tests
	@echo "🧪 Running tests..."
	@uv run pytest

format: ## 🖊️ Format code
	@echo "🖊️ Formatting code..."
	@uv run ruff format

lint: ## � Run linter
	@echo "� Running linter..."
	@uv run pyright
