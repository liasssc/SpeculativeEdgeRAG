from speculative_edge_rag.coordinator.tag_router import RuleBasedTagger, OracleTagger
from speculative_edge_rag.schemas import QueryTags, QueryTag, ServerRecord, RequestState
from speculative_edge_rag.coordinator.server_registry import ServerRegistry
from speculative_edge_rag.coordinator.server_selector import ResourceAwareServerSelector
from speculative_edge_rag.edge.evidence import EvidenceSanitizer
from speculative_edge_rag.edge.local_kg import InMemoryKnowledgeGraph, KnowledgeNode
from speculative_edge_rag.edge.retriever import LocalRetriever
from speculative_edge_rag.speculative.draft import DraftBranch, TargetDistribution
from speculative_edge_rag.speculative.multi_draft_selector import MultiDraftSelector
from speculative_edge_rag.speculative.verifier import SpeculativeVerifier

def test_tags(): assert RuleBasedTagger().tag('medical disease').level1

def test_oracle(): assert OracleTagger().tag({'level1':['legal'],'level2':['case']}).level1

def test_registry():
 r=ServerRegistry(); s=ServerRecord('s',{'legal'},{'case'}); r.register(s); assert r.get('s')==s

def test_candidates():
 r=ServerRegistry(); r.register(ServerRecord('s',{'legal'},{'case'})); assert r.candidates(QueryTags([QueryTag('legal','case',1,'oracle')]))

def test_selector():
 s=ServerRecord('s',{'legal'},{'case'}); out=ResourceAwareServerSelector().select([s],QueryTags([QueryTag('legal','case',1,'oracle')]),RequestState()); assert out.servers

def test_sanitize(): assert '[REDACTED]' in EvidenceSanitizer().sanitize('email: a@b.com')

def test_retrieve():
 kg=InMemoryKnowledgeGraph([KnowledgeNode('1','science fact',tags=('science',))]); assert LocalRetriever(kg).retrieve('science').success

def test_multidraft(): assert MultiDraftSelector().select([DraftBranch.from_tokens('s',['a'])])

def test_verify(): assert SpeculativeVerifier().verify(DraftBranch.from_tokens('s',['a']).tokens,TargetDistribution({'a':1})).accepted_tokens

def test_mock_probability_contracts():
    branch = DraftBranch.from_tokens('s', ['a', 'b', 'c', 'd'])
    assert abs(sum(branch.proposal.probabilities.values()) - 1.0) < 1e-9
    target = TargetDistribution({'a': .6, 'b': .2, 'c': .1, 'd': .1})
    assert abs(sum(target.probabilities.values()) - 1.0) < 1e-9
    result = SpeculativeVerifier().verify(branch.tokens, target)
    assert result.accepted_tokens and result.rejected_tokens
    assert abs(sum(result.residual_distribution.values()) - 1.0) < 1e-9

def test_prefix_import():
 from speculative_edge_rag.speculative.prefix import CommittedPrefix
 assert CommittedPrefix().append('a').tokens==('a',)
