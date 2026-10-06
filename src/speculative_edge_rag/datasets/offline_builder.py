"""Experimental offline partitioning; no online document movement is performed."""
from __future__ import annotations
import json, random, re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
import yaml
from .ragrouter_adapter import BoundaryChunker, TagAssigner

CHUNK_CONFIG={'default':{'chunk_size':1200,'chunk_overlap':150},'musique':{'chunk_size':700,'chunk_overlap':80},'quality':{'chunk_size':1400,'chunk_overlap':180},'ultraDomain_legal':{'chunk_size':1600,'chunk_overlap':200},'graphragBench_medical':{'chunk_size':1000,'chunk_overlap':120}}
def read_config(path): return yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
def make_chunks(docs, config=None):
    settings={**CHUNK_CONFIG,**(config or {})}; out=[]
    tagger = TagAssigner()
    for d in docs:
        s=settings.get(d.source_dataset,settings['default']); out += BoundaryChunker(int(s['chunk_size']),int(s['chunk_overlap'])).chunk_document(d, tagger)
    return out
def partition_documents(docs,chunks,config):
    servers=config.get('servers',config); servers={k:v for k,v in servers.items() if isinstance(v,dict) and k not in ('chunking','experimental_partition')}
    ep=config.get('experimental_partition',{}) if isinstance(config.get('experimental_partition',{}),dict) else {}; seed=int(config.get('seed',ep.get('seed',20260921))); replicate=bool(config.get('replicate_chunks',ep.get('replicate_chunks',False))); rng=random.Random(seed)
    chunk_level_domains=set(ep.get('chunk_level_domains',[]))
    parts={k:{'documents':[],'chunks':[]} for k in sorted(servers)}; bydoc=defaultdict(list)
    global_doc_ids=[d.doc_id for d in docs]
    if len(global_doc_ids) != len(set(global_doc_ids)):
        raise ValueError('duplicate global document ID')
    # ``doc_id`` is the adapter's global source_dataset:original_doc_id key.
    for c in chunks: bydoc[c.doc_id].append(c)
    unassigned=[]; assignments={}; chunk_assignments={}; fallback_assignment_count=0
    def domain_pool(level1_tags):
        return [sid for sid,s in sorted(servers.items()) if set(s.get('level1_tags',[]))&set(level1_tags)]
    def preferred(pool,level2_tags):
        return [sid for sid in pool if set(servers[sid].get('level2_tags',[]))&set(level2_tags)]
    def add_document_reference(sid,document):
        if not any(existing.doc_id == document.doc_id for existing in parts[sid]['documents']):
            parts[sid]['documents'].append(document)
    for d in sorted(docs,key=lambda x:x.doc_id):
        global_doc_id=d.doc_id
        pool=domain_pool(d.level1_tags)
        use_chunk_level=bool(chunk_level_domains & (set(d.level1_tags)|{d.source_dataset}))
        if use_chunk_level:
            assigned_servers=set()
            for chunk in bydoc[global_doc_id]:
                choices=preferred(pool,chunk.level2_tags)
                if not choices:
                    choices=pool; fallback_assignment_count+=bool(choices)
                if not choices:
                    unassigned.append(chunk.chunk_id); continue
                sid=min(choices,key=lambda candidate:(len(parts[candidate]['chunks']),candidate))
                parts[sid]['chunks'].append(chunk); add_document_reference(sid,d)
                assigned_servers.add(sid); chunk_assignments[chunk.chunk_id]=sid
            assignments[global_doc_id]=sorted(assigned_servers)
            continue
        choices=preferred(pool,d.level2_tags)
        if not choices:
            choices=pool; fallback_assignment_count+=bool(choices)
            chosen=[min(choices,key=lambda sid:(len(parts[sid]['documents']),sid))] if choices else []
        else:
            chosen=choices if replicate else [choices[rng.randrange(len(choices))]]
        assignments[global_doc_id]=chosen
        if not chosen: unassigned += [c.chunk_id for c in bydoc[global_doc_id]]
        for sid in chosen:
            add_document_reference(sid,d); parts[sid]['chunks'] += bydoc[global_doc_id]
            for chunk in bydoc[global_doc_id]: chunk_assignments[chunk.chunk_id]=sid
    counts=Counter(c.chunk_id for p in parts.values() for c in p['chunks']); duplicated=sorted(k for k,v in counts.items() if v>1)
    assigned_documents=sum(bool(server_ids) for server_ids in assignments.values())
    return parts,{'seed':seed,'replicate_chunks':replicate,'chunk_level_domains':sorted(chunk_level_domains),'assigned_document_count':assigned_documents,'unassigned_document_count':len(docs)-assigned_documents,'assigned_chunk_count':len(counts),'unassigned_chunk_ids':unassigned,'unassigned_chunk_count':len(unassigned),'fallback_assignment_count':fallback_assignment_count,'duplicated_chunk_ids':duplicated,'duplicated_chunk_count':sum(v-1 for v in counts.values() if v>1),'document_assignments':assignments,'chunk_assignments':chunk_assignments}
def _entities(text): return list(dict.fromkeys(re.findall(r'\b[A-Z][A-Za-z0-9\'-]*(?:\s+[A-Z][A-Za-z0-9\'-]*){0,2}\b',text)))[:12]
def build_local_kg(server_id,docs,chunks):
    nodes=[]; edges=[]; seen=set()
    def add(n):
        if n['id'] not in seen: seen.add(n['id']); nodes.append(n)
    for d in docs:
        did='document:'+d.doc_id; add({'id':did,'type':'document','text':d.title,'metadata':{'doc_id':d.doc_id,'source_dataset':d.source_dataset}})
        for level,tags in [('level1',d.level1_tags),('level2',d.level2_tags)]:
            for tag in tags: tid=f'tag:{level}:{tag}'; add({'id':tid,'type':f'tag_{level}','text':tag}); edges.append({'source':did,'target':tid,'relation':'document_has_tag'})
    for c in chunks:
        cid='chunk:'+c.chunk_id; did='document:'+c.doc_id; add({'id':cid,'type':'chunk','text':c.text,'tags':c.level2_tags,'metadata':{'chunk_id':c.chunk_id,'doc_id':c.doc_id,'chunk_index':c.chunk_index}}); edges.append({'source':did,'target':cid,'relation':'document_contains_chunk'})
        for level,tags in [('level1',c.level1_tags),('level2',c.level2_tags)]:
            for tag in tags: tid=f'tag:{level}:{tag}'; add({'id':tid,'type':f'tag_{level}','text':tag}); edges.append({'source':cid,'target':tid,'relation':'chunk_has_tag'})
        for ent in _entities(c.text): eid='entity:'+re.sub(r'\W+','-',ent.lower()).strip('-'); add({'id':eid,'type':'entity_like','text':ent}); edges.append({'source':cid,'target':eid,'relation':'chunk_mentions_entity'})
    return {'server_id':server_id,'visibility':'private','nodes':nodes,'edges':edges}
def capability_entry(sid,docs,chunks):
    l1=Counter(t for c in chunks for t in c.level1_tags); l2=Counter(t for c in chunks for t in c.level2_tags)
    ratios={tag:count/len(chunks) for tag,count in sorted(l2.items())} if chunks else {}
    return {'server_id':sid,'level1_tags':sorted(l1),'level2_tags':sorted(l2),'document_count':len(docs),'chunk_count':len(chunks),'tag_coverage_statistics':{'level1_chunk_counts':dict(sorted(l1.items())),'level2_chunk_counts':dict(sorted(l2.items())),'level2_coverage_ratios':ratios},'quality':{'status':'not_evaluated','score':None},'resource_state':{'status':'unknown','available':True,'queue_length':0,'load':0.0}}
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row.to_dict() if hasattr(row,'to_dict') else row,ensure_ascii=False,sort_keys=True)+'\n')
def write_offline_dataset(processed_dir,edge_dir,coordinator_dir,docs,queries,chunks,parts,stats,config):
    p=Path(processed_dir); e=Path(edge_dir); c=Path(coordinator_dir); write_jsonl(p/'documents.jsonl',docs); write_jsonl(p/'chunks.jsonl',chunks); write_jsonl(p/'queries.jsonl',queries)
    registry=[]
    for sid,part in parts.items():
        d=e/sid; d.mkdir(parents=True,exist_ok=True); write_jsonl(d/'documents.jsonl',part['documents']); write_jsonl(d/'chunks.jsonl',part['chunks']); (d/'local_kg.json').write_text(json.dumps(build_local_kg(sid,part['documents'],part['chunks']),ensure_ascii=False,indent=2),encoding='utf-8'); (d/'metadata.json').write_text(json.dumps({'server_id':sid,'partition_kind':'experimental_offline_partition','online_routing_operation':False,'seed':stats['seed'],'replicate_chunks':stats['replicate_chunks'],'document_reference_count':len(part['documents']),'document_count':len(part['documents']),'chunk_count':len(part['chunks'])},ensure_ascii=False,indent=2),encoding='utf-8'); registry.append(capability_entry(sid,part['documents'],part['chunks']))
    c.mkdir(parents=True,exist_ok=True); (c/'server_capability_registry.json').write_text(json.dumps({'schema_version':1,'servers':registry},ensure_ascii=False,indent=2),encoding='utf-8'); (p/'partition_stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
