"""Build the model-free experimental edge-server dataset partitions."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from speculative_edge_rag.datasets.ragrouter_adapter import RAGRouterDatasetAdapter
from speculative_edge_rag.datasets.offline_builder import make_chunks, partition_documents, read_config, write_offline_dataset

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', default='data/raw/ragrouter_bench')
    ap.add_argument('--partition-config', default='configs/server_partition.yaml')
    ap.add_argument('--processed-dir', default='data/processed')
    ap.add_argument('--edge-dir', default='data/edge_servers')
    ap.add_argument('--coordinator-dir', default='data/coordinator')
    ap.add_argument('--limit-per-dataset', type=int, default=None)
    a = ap.parse_args()
    docs, queries = RAGRouterDatasetAdapter().load(a.input_dir, a.limit_per_dataset)
    config = read_config(a.partition_config)
    chunks = make_chunks(docs, config.get('chunking'))
    parts, stats = partition_documents(docs, chunks, config)
    write_offline_dataset(a.processed_dir, a.edge_dir, a.coordinator_dir, docs, queries, chunks, parts, stats, config)
    print(f'built {len(docs)} documents, {len(chunks)} chunks, {len(queries)} queries across {len(parts)} servers')

if __name__ == '__main__':
    main()
