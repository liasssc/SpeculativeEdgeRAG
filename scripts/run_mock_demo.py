import sys; sys.path.insert(0, 'src')
from speculative_edge_rag.coordinator import *
from speculative_edge_rag.schemas import ServerRecord, RequestState
from speculative_edge_rag.edge import *
from speculative_edge_rag.coordinator.coordinator import Coordinator

def main():
    reg=ServerRegistry(); edges={}
    configs = [
        ('edge-1', {'encyclopedic'}, {'science','history'}, [('sci-1','Science explains evidence and history.',('science','history'))]),
        ('edge-2', {'medical'}, {'disease','risk'}, [('med-1','Disease risk evidence supports medical analysis.',('disease','risk'))]),
        ('edge-3', {'legal'}, {'statute','liability'}, [('law-1','Statute and liability define legal obligations.',('statute','liability'))]),
    ]
    for sid, level1, level2, nodes in configs:
        reg.register(ServerRecord(sid, level1, level2, 0.8, 0, 0.2, 0.1, 20, 30, 10, True))
        edges[sid]=EdgeServer(sid, InMemoryKnowledgeGraph([KnowledgeNode(nid,text,tags) for nid,text,tags in nodes]))
    c=Coordinator(RuleBasedTagger(),reg,ResourceAwareServerSelector(max_servers=4),edges)
    result = c.process_request('science history disease risk statute liability', RequestState(request_id='demo', query_id='query-demo', deadline_ms=500, max_servers=4))
    print(result)
if __name__=='__main__': main()
