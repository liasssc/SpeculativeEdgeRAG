# SpeculativeEdgeRAG

Python 3.11+ baseline framework for multi edge server knowledge tag routing, resource aware selection, secure evidence capsules, and multi draft speculative decoding. All model and retrieval components are mocks with replaceable interfaces.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
python scripts/run_mock_demo.py
```

Core modules are separated under `src/speculative_edge_rag/{coordinator,edge,speculative}`. `RAGRouterBenchAdapter` is the integration point for future datasets; replace `RuleBasedTagger`, `LocalRetriever/InMemoryKnowledgeGraph`, and `EdgeServer` wiring when adding real data. Cloud model integration belongs behind `TargetDistribution` and `SpeculativeVerifier`.

## RAGRouter-Bench first-stage data pipeline

Place either the RAGRouter-Bench repository files or Hugging Face exports under
`data/raw/ragrouter_bench/` (the adapter accepts `Corpus.json`, `Question.json`
and common JSON/JSONL variants). No dataset content is embedded in source code.

```bash
python scripts/prepare_ragrouter_data.py --input-dir data/raw/ragrouter_bench --output-dir data/processed
python scripts/build_local_graphs.py --processed-dir data/processed --partition-config configs/server_partition.yaml --output-dir data/knowledge_graphs
python scripts/inspect_ragrouter_data.py
```

The taxonomy in `configs/tag_taxonomy.yaml` assigns each document and query a
fixed level-one domain and multiple explainable keyword-based level-two tags.
`configs/server_partition.yaml` maps those tags to eight private edge-server
partitions. Each server receives only matching documents and gets an independent
JSON knowledge graph containing document, chunk, entity-like, and tag nodes.

This stage does not download or load Qwen, edge draft, or cloud target models,
and does not change the multi-draft or acceptance algorithms. The next stage is
planned to integrate Qwen2.5-1.5B and Qwen2.5-7B.

## Current mock verifier scope

The verifier is a **deterministic mock baseline** using `p >= q` and normalized
residual correction. It has not implemented randomized `min(1,p/q)` acceptance,
first rejection position handling within a block, true residual sampling, or
committed prefix rollback.
