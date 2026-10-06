"""Validate the shape and privacy boundaries of an offline partition build."""
from __future__ import annotations
import argparse, json
from pathlib import Path

ALLOWED_REGISTRY = {'server_id','level1_tags','level2_tags','document_count','chunk_count','tag_coverage_statistics','quality','resource_state'}
REQUIRED_FILES = {'documents.jsonl','chunks.jsonl','local_kg.json','metadata.json'}

def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def validate(root: Path) -> None:
    edge = root / 'edge_servers'; coord = root / 'coordinator'; processed = root / 'processed'
    assert (processed/'queries.jsonl').exists(), 'queries must be coordinator-side processed data'
    registry = json.loads((coord/'server_capability_registry.json').read_text(encoding='utf-8'))
    assert set(registry) == {'schema_version','servers'}
    for entry in registry['servers']:
        assert set(entry) <= ALLOWED_REGISTRY
        assert not any(k in entry for k in ('text','contents','supporting_facts','local_kg'))
        server = edge / entry['server_id']; assert REQUIRED_FILES <= {p.name for p in server.iterdir()}
        assert not (server/'queries.jsonl').exists()
        docs, chunks = rows(server/'documents.jsonl'), rows(server/'chunks.jsonl')
        kg = json.loads((server/'local_kg.json').read_text(encoding='utf-8'))
        assert len(docs) == entry['document_count'] and len(chunks) == entry['chunk_count']
        chunk_ids = {f"chunk:{chunk['chunk_id']}" for chunk in chunks}
        contained = {edge['target'] for edge in kg.get('edges', []) if edge.get('relation') == 'document_contains_chunk'}
        assert contained == chunk_ids, 'local KG must only reference local chunks'
        expected_l2 = {}
        for chunk in chunks:
            for tag in chunk.get('level2_tags', []): expected_l2[tag] = expected_l2.get(tag, 0) + 1
        coverage = entry['tag_coverage_statistics']
        assert coverage['level2_chunk_counts'] == dict(sorted(expected_l2.items()))
        expected_ratios = {tag: count / len(chunks) for tag, count in sorted(expected_l2.items())} if chunks else {}
        assert coverage['level2_coverage_ratios'] == expected_ratios
        relations = {e.get('relation') for e in kg.get('edges', [])}
        if chunks:
            assert {'document_contains_chunk','document_has_tag','chunk_has_tag'} <= relations
    print(f"validated {len(registry['servers'])} offline edge-server partitions")

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('root', help='build root containing processed/, edge_servers/, coordinator/')
    validate(Path(ap.parse_args().root))
