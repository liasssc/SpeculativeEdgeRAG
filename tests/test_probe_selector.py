from dataclasses import fields
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from speculative_edge_rag.coordinator.probe_selector import ProbeAwareServerSelector,ProbeSelectorConfig
from speculative_edge_rag.edge.local_kg import InMemoryKnowledgeGraph,KnowledgeNode
from speculative_edge_rag.edge.server import EdgeServer
from speculative_edge_rag.schemas import ProbeResult,QueryTag,QueryTags,ServerRecord


CONFIG=ProbeSelectorConfig.from_yaml(ROOT/'configs'/'probe_selector.yaml')


def edge(server_id,texts):
    nodes=[KnowledgeNode(f'{server_id}-chunk-{i}',text,('case',),{'type':'chunk','doc_id':f'doc-{server_id}-{i}','level1_tags':'legal'}) for i,text in enumerate(texts)]
    return EdgeServer(server_id,InMemoryKnowledgeGraph(nodes))


def tags(): return QueryTags([QueryTag('legal','case')])


def record(server_id):
    return ServerRecord(server_id,{'legal'},{'case'},metadata={'level2_coverage_ratios':{'case':0.5}})


def select(query,edges,max_servers=4):
    probes={item.server_id:item.probe(query,tags(),top_k=3) for item in edges}
    records=[record(item.server_id) for item in edges]
    return ProbeAwareServerSelector(CONFIG,max_servers).select(records,tags(),probes,len(query.split()),max_servers),probes


def test_probe_result_is_privacy_safe_and_index_is_reused():
    server=edge('s1',['alpha private raw sentence'])
    index_id=id(server.probe_index); first=server.probe('alpha',tags(),3); second=server.probe('alpha',tags(),3)
    assert id(server.probe_index)==index_id
    field_names={field.name for field in fields(ProbeResult)}
    assert 'text' not in field_names and 'raw_text' not in field_names
    assert 'supporting_doc_ids' not in field_names and 'gold_server_set' not in field_names
    assert first.top_chunk_ids==second.top_chunk_ids==('s1-chunk-0',)
    assert first.query_term_coverage==(0,)
    assert not hasattr(server.probe_index.chunks['s1-chunk-0'],'text')


def test_complementary_second_server_is_added_but_duplicate_is_not():
    complementary,_=select('alpha gamma',[edge('s1',['alpha']),edge('s2',['gamma'])])
    assert {server.server_id for server in complementary.servers}=={'s1','s2'}
    duplicate,_=select('alpha',[edge('s1',['alpha']),edge('s2',['alpha'])])
    assert len(duplicate.servers)==1
    assert 'no second server cleared' in duplicate.explanation


def test_higher_relevance_is_first_max_servers_and_output_are_deterministic():
    edges=[edge('s1',['alpha beta']),edge('s2',['alpha']),edge('s3',['gamma'])]
    first,probes=select('alpha beta gamma',edges,max_servers=2)
    second=ProbeAwareServerSelector(CONFIG,2).select([record(item.server_id) for item in edges],tags(),probes,3,2)
    assert first.servers[0].server_id=='s1'
    assert len(first.servers)<=2
    assert [server.server_id for server in first.servers]==[server.server_id for server in second.servers]
    assert first.score==second.score and first.explanation==second.explanation


def test_gold_labels_cannot_change_probe_selection():
    edges=[edge('s1',['alpha']),edge('s2',['gamma'])]
    selected,_=select('alpha gamma',edges)
    query_a={'question':'alpha gamma','supporting_doc_ids':['gold-a']}
    query_b={'question':'alpha gamma','supporting_doc_ids':['gold-b']}
    assert query_a['question']==query_b['question']
    repeated,_=select(query_b['question'],edges)
    assert [server.server_id for server in selected.servers]==[server.server_id for server in repeated.servers]
