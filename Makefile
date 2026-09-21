.PHONY: help test lint budget sim phase0 phase0-signal phase0-anchors data clean
PY := .venv/bin/python

help:
	@grep -E '^[a-z0-9-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | expand -t22

test:            ## unit tests: seconds, no weights, no network
	$(PY) -m pytest tests/ -q

lint:            ## ruff
	.venv/bin/ruff check src tests scripts

budget:          ## the token arithmetic that drove the slate design
	$(PY) -m frontierrank.core.budget
	@echo
	$(PY) -c "from transformers import AutoTokenizer as T; from frontierrank.models.laya.budget_probe import probe_table; print(probe_table(T.from_pretrained('data/cache/laya/tokenizer')))"

sim:             ## the numpy-only simulation from plan.md Part VIII
	$(PY) scripts/sim_smoke_test.py

phase0:          ## THE GATE: is c_S a per-slate shift on real text?
	$(PY) scripts/run_phase0.py --dataset trec-covid --queries 12

phase0-signal:   ## does the checkpoint have signal, or did we starve it?
	$(PY) scripts/run_phase0_signal.py --dataset trec-covid --queries 10

phase0-anchors:  ## falsification test 4: does anchoring buy the global scale?
	$(PY) scripts/run_phase0_anchors.py --dataset trec-covid --queries 10

data:            ## BEIR corpora into data/raw
	@mkdir -p data/raw
	@for ds in nfcorpus scifact trec-covid; do \
	  test -d data/raw/$$ds || ( echo "-- $$ds" && \
	  curl -s -o /tmp/$$ds.zip "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$$ds.zip" && \
	  unzip -q -o /tmp/$$ds.zip -d data/raw && rm /tmp/$$ds.zip ); done
	@echo "ready:" && ls data/raw

clean:           ## drop run artifacts, keep manifests and metrics
	find runs -name '*.npz' -delete

phase0-signal-full: ## F7 at full scale with paired bootstrap CIs
	$(PY) scripts/run_phase0_signal.py --dataset trec-covid --queries 50 --slates-per-query 30

anchor-probe:    ## are the templated pivots actually pivots?
	$(PY) scripts/run_anchor_probe.py --dataset trec-covid --queries 20

train:           ## Phase 1: fine-tune on nfcorpus, evaluate on trec-covid
	$(PY) scripts/run_phase1_train.py --train nfcorpus --eval trec-covid

check-first-stage: ## validate BM25 against published BEIR numbers
	$(PY) scripts/check_first_stage.py
