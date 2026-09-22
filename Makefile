.PHONY: help install weights quickstart test lint budget example-data findings clean
PY := .venv/bin/python

help:            ## show this
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t20

# ---------------------------------------------------------------- the recipe
install:         ## install the package and training deps
	uv venv --python 3.12 .venv || true
	uv pip install -e '.[train,dev]'

weights:         ## fetch the Laya checkpoint (~1.7 GB)
	$(PY) -c "from huggingface_hub import snapshot_download as d; \
	d('convaiinnovations/laya', local_dir='data/cache/laya', \
	allow_patterns=['model.safetensors','rl_agent_config.json','encoder/*','tokenizer/*'])"

quickstart:      ## fine-tune end to end in ~1 minute. START HERE.
	$(PY) examples/01_quickstart.py

budget:          ## the token arithmetic that governs question design
	$(PY) -m systemone.model.budget
	@echo
	$(PY) -c "from transformers import AutoTokenizer as T; \
	from systemone.model.budget_probe import probe_table; \
	print(probe_table(T.from_pretrained('data/cache/laya/tokenizer')))"

test:            ## unit tests: seconds, no weights, no network
	$(PY) -m pytest tests/unit -q

lint:
	.venv/bin/ruff check src tests examples scripts

# ------------------------------------------------- the reranking worked example
example-data:    ## BEIR corpora for the reranking example
	@mkdir -p data/raw
	@for ds in nfcorpus scifact trec-covid; do \
	  test -d data/raw/$$ds || ( echo "-- $$ds" && \
	  curl -s -o /tmp/$$ds.zip "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$$ds.zip" && \
	  unzip -q -o /tmp/$$ds.zip -d data/raw && rm /tmp/$$ds.zip ); done
	@ls data/raw

example-train:   ## reproduce the reranking fine-tune
	$(PY) scripts/reranking/run_phase1_train.py --train nfcorpus --eval trec-covid

example-baselines: ## the honest competition: MiniLM, Qwen3, BM25
	$(PY) scripts/reranking/run_baselines.py --only minilm,laya --queries 30
	$(PY) scripts/reranking/run_baselines.py --only qwen --queries 30

example-gate:    ## the Phase 0 identifiability gate
	$(PY) scripts/reranking/run_phase0.py --dataset trec-covid --queries 12

example-first-stage: ## validate BM25 against published BEIR numbers
	$(PY) scripts/reranking/check_first_stage.py

findings:        ## what we measured, including what did not work
	@echo "see FINDINGS.md, RECIPE.md and docs/ROADMAP.md"

clean:
	find runs -name '*.npz' -delete
