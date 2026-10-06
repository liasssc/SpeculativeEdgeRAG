from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))

from analyze_query_complementarity import (
    aggregate,
    analyze_query,
    current_selector_choice,
    method_evaluation,
)
from speculative_edge_rag.coordinator.server_registry import ServerRegistry
from speculative_edge_rag.coordinator.server_selector import ResourceAwareServerSelector
from speculative_edge_rag.schemas import ServerRecord


def synthetic_registry():
    registry=ServerRegistry()
    servers=[
        ServerRecord('s1',{'legal'},{'case'}),
        ServerRecord('s2',{'legal'},{'remedy'}),
        ServerRecord('s3',{'literature'},{'character'}),
    ]
    for server in servers: registry.register(server)
    capabilities={server.server_id:{'level1_tags':sorted(server.level1_tags),'level2_tags':sorted(server.level2_tags)} for server in servers}
    return registry,capabilities


def query(query_id,supporting):
    return {
        'query_id':query_id,
        'source_dataset':'ultraDomain_legal',
        'query_type':'multi_hop',
        'level1_tags':['legal'],
        'level2_tags':['case'],
        'supporting_doc_ids':supporting,
    }


def test_single_and_two_server_gold_evidence():
    registry,capabilities=synthetic_registry(); selector=ResourceAwareServerSelector()
    doc_servers={'d1':{'s1'},'d2':{'s2'}}
    single=analyze_query(query('single',['d1']),registry,capabilities,doc_servers,selector)
    assert single['minimum_required_server_count']==1
    assert single['methods']['current_selector']['supporting_document_recall']==1.0
    assert single['methods']['single_server_oracle']['selected_servers']==['s1']

    complementary=analyze_query(query('two',['d1','d2']),registry,capabilities,doc_servers,selector)
    assert complementary['minimum_required_server_count']==2
    assert complementary['gold_server_set']==['s1','s2']
    assert complementary['methods']['current_selector']['selected_servers']==['s1']
    assert complementary['methods']['current_selector']['supporting_document_recall']==0.5
    assert complementary['methods']['current_selector']['misses_required_evidence']
    assert complementary['methods']['gold_server_oracle']['supporting_document_recall']==1.0
    metrics=aggregate([complementary])['methods']['current_selector']
    assert metrics['multi_server_required_query_recall']==0.0


def test_second_server_without_new_evidence_has_zero_gain():
    evaluation=method_evaluation(['s1','s2'],['d1'],{'d1':{'s1','s2'}},{'s1','s2'})
    assert evaluation['supporting_document_recall']==1.0
    assert evaluation['second_server_marginal_supporting_document_gain']==0


def test_current_selector_choice_is_independent_of_gold_labels():
    registry,_=synthetic_registry(); selector=ResourceAwareServerSelector()
    candidates_a,selected_a=current_selector_choice('a',['legal'],['case'],registry,selector)
    candidates_b,selected_b=current_selector_choice('b',['legal'],['case'],registry,selector)
    assert candidates_a==candidates_b==['s1','s2']
    assert selected_a==selected_b==['s1']
