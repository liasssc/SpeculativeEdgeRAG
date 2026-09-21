#!/usr/bin/env python3
"""Report parsed raw RAGRouter records without changing the source files."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from speculative_edge_rag.datasets.ragrouter_adapter import RAGRouterDatasetAdapter


def fact_type(value):
    if isinstance(value, list):
        if not value: return 'empty list'
        kinds = {type(item).__name__ for item in value}
        return 'list[' + ','.join(sorted(kinds)) + ']'
    return type(value).__name__


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='data/raw/ragrouter_bench')
    args = parser.parse_args()
    adapter = RAGRouterDatasetAdapter()
    docs, queries = adapter.load(args.input_dir)
    for domain in ('musique', 'quality', 'ultraDomain_legal', 'graphragBench_medical'):
        domain_docs = [d for d in docs if d.source_dataset == domain]
        domain_queries = [q for q in queries if q.source_dataset == domain]
        report = {
            'domain': domain,
            'documents': len(domain_docs),
            'queries': len(domain_queries),
            'first_document': vars(domain_docs[0]) if domain_docs else None,
            'first_query': vars(domain_queries[0]) if domain_queries else None,
            'parse_failures': adapter.parse_failures.get(domain, {'documents': 0, 'queries': 0}),
            'supporting_facts_types': dict(Counter(fact_type(q.supporting_facts) for q in domain_queries)),
        }
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__': main()
