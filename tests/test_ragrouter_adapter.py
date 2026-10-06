import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from speculative_edge_rag.datasets.ragrouter_adapter import RAGRouterDatasetAdapter


def test_json_array_and_wrappers():
    adapter = RAGRouterDatasetAdapter()
    doc = adapter.parse_documents([{'doc_id': 3, 'contents': 'content'}], 'quality')[0]
    assert doc.doc_id == 'quality:3'
    assert doc.metadata['original_doc_id'] == '3'
    assert adapter.parse_documents({'records': [{'title': 'T', 'document': 'body'}]}, 'quality')[0].text == 'body'
    query = adapter.parse_queries({'data': [{'question_id': 'q', 'query': 'why', 'type': 'multi', 'supporting_facts': ['sentence']}]}, 'quality')[0]
    assert query.supporting_facts == ['sentence']
    assert query.supporting_doc_ids == []
    assert query.metadata['supporting_doc_ids_source'] == 'unavailable'


def test_jsonl_and_medical_single_object(tmp_path):
    domain = tmp_path / 'graphragBench_medical'
    domain.mkdir()
    (domain / 'Corpus.json').write_text(json.dumps({'id': 7, 'title': 'Medical', 'context': 'cancer treatment'}))
    (domain / 'Question.json').write_text('\n'.join([
        json.dumps({'id': 'q1', 'question': 'What treatment?', 'answer': 'surgery', 'type': 'single_hop', 'supporting_facts': ['A supporting sentence.']}),
        json.dumps({'id': 'q2', 'text': 'What test?', 'supporting_facts': [{'doc_id': 7, 'text': 'Evidence'}]}),
        '{bad json',
    ]))
    adapter = RAGRouterDatasetAdapter()
    with pytest.warns(RuntimeWarning,match='unrecoverable JSON record'):
        docs, queries = adapter.load(tmp_path)
    assert (docs[0].doc_id, docs[0].title, docs[0].text) == ('graphragBench_medical:7', 'Medical', 'cancer treatment')
    assert len(queries) == 2
    assert queries[0].query_type == 'single_hop'
    assert queries[0].supporting_facts == ['A supporting sentence.']
    assert queries[0].supporting_doc_ids == []
    assert queries[1].supporting_facts == [{'doc_id': 7, 'text': 'Evidence'}]
    assert queries[1].supporting_doc_ids == ['graphragBench_medical:7']
    assert queries[1].metadata['supporting_doc_ids_source'] == 'derived_from_supporting_facts'
    assert adapter.parse_failures['graphragBench_medical']['queries'] == 1
    assert adapter.parse_errors[0]['start_line'] == 3


def test_bounded_multiline_jsonl_recovery_and_explicit_support_priority(tmp_path):
    domain = tmp_path / 'ultraDomain_legal'
    domain.mkdir()
    (domain / 'Corpus.json').write_text(
        '{"title": "", "context": "Exhibit 10.17 first half '
        '\nDocks:\\nsecond half", "id": 29}\n',
        encoding='utf-8',
    )
    (domain / 'Question.json').write_text(json.dumps({
        'id': 'q',
        'question': 'Which case?',
        'supporting_doc_ids': [29, 29],
        'supporting_facts': [{'doc_id': 30, 'text': 'lower priority'}],
    }),encoding='utf-8')
    adapter = RAGRouterDatasetAdapter()
    docs, queries = adapter.load(tmp_path)
    assert len(docs) == 1
    assert docs[0].doc_id == 'ultraDomain_legal:29'
    assert 'first half \nDocks:' in docs[0].text
    assert adapter.recovered_records['ultraDomain_legal']['documents'] == 1
    assert adapter.parse_errors == []
    assert queries[0].supporting_doc_ids == ['ultraDomain_legal:29']
    assert queries[0].metadata['supporting_doc_ids_source'] == 'explicit'
