import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from speculative_edge_rag.datasets.offline_builder import (
    build_local_kg,
    capability_entry,
    make_chunks,
    partition_documents,
    write_offline_dataset,
)
from speculative_edge_rag.datasets.ragrouter_adapter import ChunkRecord, RAGRouterDatasetAdapter


def test_duplicate_source_doc_ids_remain_isolated(tmp_path):
    adapter = RAGRouterDatasetAdapter()
    shared_text = 'A cancer disease and court case.'
    medical = adapter.parse_documents(
        [{'doc_id': 0, 'title': 'Cancer', 'text': shared_text}],
        'graphragBench_medical',
    )[0]
    legal = adapter.parse_documents(
        [{'doc_id': 0, 'title': 'Court', 'text': shared_text}],
        'ultraDomain_legal',
    )[0]
    medical_query = adapter.parse_queries(
        [{'id': 'm', 'question': 'Which disease?', 'supporting_doc_ids': [0]}],
        'graphragBench_medical',
    )[0]
    legal_query = adapter.parse_queries(
        [{'id': 'l', 'question': 'Which case?', 'supporting_doc_ids': [0]}],
        'ultraDomain_legal',
    )[0]

    docs = [medical, legal]
    assert {doc.doc_id for doc in docs} == {
        'graphragBench_medical:0',
        'ultraDomain_legal:0',
    }
    assert {doc.metadata['original_doc_id'] for doc in docs} == {'0'}
    assert medical_query.supporting_doc_ids == ['graphragBench_medical:0']
    assert legal_query.supporting_doc_ids == ['ultraDomain_legal:0']

    chunks = make_chunks(docs)
    assert len({chunk.chunk_id for chunk in chunks}) == 2
    assert {chunk.doc_id for chunk in chunks} == {doc.doc_id for doc in docs}

    config = {
        'medical-server': {'level1_tags': ['medical'], 'level2_tags': ['disease']},
        'legal-server': {'level1_tags': ['legal'], 'level2_tags': ['case']},
    }
    parts, stats = partition_documents(docs, chunks, config)
    assert [doc.doc_id for doc in parts['medical-server']['documents']] == [medical.doc_id]
    assert [chunk.doc_id for chunk in parts['medical-server']['chunks']] == [medical.doc_id]
    assert [doc.doc_id for doc in parts['legal-server']['documents']] == [legal.doc_id]
    assert [chunk.doc_id for chunk in parts['legal-server']['chunks']] == [legal.doc_id]
    assert set(stats['document_assignments']) == {medical.doc_id, legal.doc_id}

    write_offline_dataset(
        tmp_path / 'processed',
        tmp_path / 'edges',
        tmp_path / 'coordinator',
        docs,
        [medical_query, legal_query],
        chunks,
        parts,
        stats,
        config,
    )
    registry = json.loads(
        (tmp_path / 'coordinator' / 'server_capability_registry.json').read_text()
    )
    written_docs = [
        json.loads(line)
        for line in (tmp_path / 'processed' / 'documents.jsonl').read_text().splitlines()
    ]
    assert {doc['doc_id'] for doc in written_docs} == {medical.doc_id, legal.doc_id}
    servers = {server['server_id']: server for server in registry['servers']}
    assert servers['medical-server']['level1_tags'] == ['medical']
    assert (servers['medical-server']['document_count'], servers['medical-server']['chunk_count']) == (1, 1)
    assert servers['legal-server']['level1_tags'] == ['legal']
    assert (servers['legal-server']['document_count'], servers['legal-server']['chunk_count']) == (1, 1)


def test_level1_fallback_assigns_every_document_and_chunk():
    adapter = RAGRouterDatasetAdapter()
    docs = adapter.parse_documents([
        {'doc_id': 1, 'text': 'No matching subtype words.'},
        {'doc_id': 2, 'text': 'Still no matching subtype words.'},
    ],'musique')
    chunks = make_chunks(docs)
    config = {
        'edge-a': {'level1_tags': ['encyclopedic'], 'level2_tags': ['person']},
        'edge-b': {'level1_tags': ['encyclopedic'], 'level2_tags': ['science']},
    }
    parts, stats = partition_documents(docs, chunks, config)
    assert stats['assigned_document_count'] == 2
    assert stats['assigned_chunk_count'] == 2
    assert stats['unassigned_document_count'] == 0
    assert stats['unassigned_chunk_count'] == 0
    assert stats['fallback_assignment_count'] == 2
    assert [len(parts[sid]['documents']) for sid in ('edge-a','edge-b')] == [1,1]


def test_chunk_level_partition_keeps_kg_and_registry_local(tmp_path):
    adapter = RAGRouterDatasetAdapter()
    document = adapter.parse_documents(
        [{'doc_id': 0, 'title': 'Medical', 'text': 'medical'}],
        'graphragBench_medical',
    )[0]
    chunks = [
        ChunkRecord('c-disease',document.doc_id,0,'cancer disease','medical',document.source_dataset,['medical'],['disease']),
        ChunkRecord('c-treatment',document.doc_id,1,'surgery treatment','medical',document.source_dataset,['medical'],['treatment']),
        ChunkRecord('c-summary',document.doc_id,2,'overview','medical',document.source_dataset,['medical'],['summary']),
    ]
    config = {
        'experimental_partition': {'chunk_level_domains': ['medical']},
        'edge-3': {'level1_tags': ['medical'], 'level2_tags': ['disease']},
        'edge-4': {'level1_tags': ['medical'], 'level2_tags': ['treatment']},
    }
    parts, stats = partition_documents([document],chunks,config)
    assert parts['edge-3']['chunks'] and parts['edge-4']['chunks']
    assert stats['document_assignments'][document.doc_id] == ['edge-3','edge-4']
    assert stats['assigned_document_count'] == 1
    assert stats['assigned_chunk_count'] == 3
    assert stats['unassigned_document_count'] == 0
    assert stats['unassigned_chunk_count'] == 0
    assert stats['fallback_assignment_count'] == 1
    assert set(stats['chunk_assignments']) == {chunk.chunk_id for chunk in chunks}

    for sid,part in parts.items():
        kg = build_local_kg(sid,part['documents'],part['chunks'])
        local_chunk_nodes = {f"chunk:{chunk.chunk_id}" for chunk in part['chunks']}
        contained = {edge['target'] for edge in kg['edges'] if edge['relation'] == 'document_contains_chunk'}
        assert contained == local_chunk_nodes
        registry = capability_entry(sid,part['documents'],part['chunks'])
        expected = {}
        for chunk in part['chunks']:
            for tag in chunk.level2_tags: expected[tag] = expected.get(tag,0)+1
        assert registry['tag_coverage_statistics']['level2_chunk_counts'] == expected
        assert registry['tag_coverage_statistics']['level2_coverage_ratios'] == {
            tag: count / len(part['chunks']) for tag,count in expected.items()
        }

    write_offline_dataset(tmp_path/'processed',tmp_path/'edges',tmp_path/'coordinator',[document],[],chunks,parts,stats,config)
    written_stats = json.loads((tmp_path/'processed'/'partition_stats.json').read_text())
    assert written_stats['document_assignments'][document.doc_id] == ['edge-3','edge-4']
    assert len(written_stats['chunk_assignments']) == 3
