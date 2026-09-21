import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from speculative_edge_rag.datasets.ragrouter_adapter import RAGRouterDatasetAdapter


def test_json_array_and_wrappers():
    adapter = RAGRouterDatasetAdapter()
    assert adapter.parse_documents([{'doc_id': 3, 'contents': 'content'}], 'quality')[0].doc_id == '3'
    assert adapter.parse_documents({'records': [{'title': 'T', 'document': 'body'}]}, 'quality')[0].text == 'body'
    assert adapter.parse_queries({'data': [{'question_id': 'q', 'query': 'why', 'type': 'multi', 'supporting_facts': ['sentence']}]}, 'quality')[0].supporting_facts == ['sentence']


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
    docs, queries = adapter.load(tmp_path)
    assert (docs[0].doc_id, docs[0].title, docs[0].text) == ('7', 'Medical', 'cancer treatment')
    assert len(queries) == 2
    assert queries[0].query_type == 'single_hop'
    assert queries[0].supporting_facts == ['A supporting sentence.']
    assert queries[0].supporting_doc_ids == []
    assert queries[1].supporting_facts == [{'doc_id': 7, 'text': 'Evidence'}]
    assert adapter.parse_failures['graphragBench_medical']['queries'] == 1
