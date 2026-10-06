#!/usr/bin/env python3
"""Calibrate and evaluate the privacy-safe probe-aware selector."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'src'; SCRIPTS=ROOT/'scripts'
for path in (SRC,SCRIPTS):
    if str(path) not in sys.path: sys.path.insert(0,str(path))

from analyze_query_complementarity import current_selector_choice,greedy_gold_order,load_partition_indexes,method_evaluation,minimum_cover,records
from speculative_edge_rag.coordinator.probe_selector import ProbeAwareServerSelector,ProbeSelectorConfig
from speculative_edge_rag.coordinator.server_registry import ServerRegistry
from speculative_edge_rag.coordinator.server_selector import ResourceAwareServerSelector
from speculative_edge_rag.coordinator.tag_router import OracleTagger
from speculative_edge_rag.edge.local_kg import InMemoryKnowledgeGraph,KnowledgeNode
from speculative_edge_rag.edge.probe import normalize_terms
from speculative_edge_rag.edge.server import EdgeServer
from speculative_edge_rag.schemas import ServerRecord


METHODS=("current_selector","probe_aware_selector","single_server_oracle","all_domain_servers","gold_server_oracle")


def load_registry(path: Path) -> tuple[ServerRegistry,dict[str,dict[str,Any]]]:
    payload=json.loads(path.read_text(encoding='utf-8')); registry=ServerRegistry(); capabilities={}
    for raw in payload['servers']:
        state=raw.get('resource_state') or {}; stats=raw.get('tag_coverage_statistics') or {}; sid=str(raw['server_id'])
        server=ServerRecord(sid,set(raw.get('level1_tags') or []),set(raw.get('level2_tags') or []),queue_length=int(state.get('queue_length',0)),load=float(state.get('load',0.0)),available=bool(state.get('available',state.get('status')!='offline')),metadata={'level2_coverage_ratios':stats.get('level2_coverage_ratios',{}),'document_count':raw.get('document_count',0),'chunk_count':raw.get('chunk_count',0)})
        registry.register(server); capabilities[sid]={'level1_tags':sorted(server.level1_tags),'level2_tags':sorted(server.level2_tags)}
    return registry,capabilities


def load_edge_servers(edge_dir: Path) -> dict[str,EdgeServer]:
    servers={}
    for directory in sorted(edge_dir.iterdir()):
        chunk_path=directory/'chunks.jsonl'
        if not chunk_path.exists(): continue
        nodes=[KnowledgeNode(str(chunk['chunk_id']),str(chunk.get('text','')),tuple(map(str,chunk.get('level2_tags') or [])),{'type':'chunk','doc_id':str(chunk.get('doc_id','')),'level1_tags':','.join(map(str,chunk.get('level1_tags') or []))}) for chunk in records(chunk_path)]
        servers[directory.name]=EdgeServer(directory.name,InMemoryKnowledgeGraph(nodes))
    return servers


def split_name(query_id: str) -> str:
    return 'calibration' if int(hashlib.sha256(query_id.encode()).hexdigest()[:8],16)%100<20 else 'evaluation'


def prepare_case(query: dict[str,Any],registry: ServerRegistry,capabilities: dict[str,dict[str,Any]],edge_servers: dict[str,EdgeServer],doc_servers: dict[str,set[str]],baseline: ResourceAwareServerSelector,config: ProbeSelectorConfig,max_servers: int) -> dict[str,Any]:
    public={'query_id':str(query.get('query_id','')),'question':str(query.get('question','')),'source_dataset':str(query.get('source_dataset','')),'query_type':str(query.get('query_type','')),'level1_tags':list(query.get('level1_tags') or []),'level2_tags':list(query.get('level2_tags') or [])}
    tags=OracleTagger().tag({'level1':public['level1_tags'],'level2':public['level2_tags']}); candidates=registry.candidates(tags)
    probes={server.server_id:edge_servers[server.server_id].probe(public['question'],tags,config.top_k) for server in candidates}
    _,current=current_selector_choice(public['query_id'],public['level1_tags'],public['level2_tags'],registry,baseline,max_servers=max_servers)
    gold_docs=list(dict.fromkeys(map(str,query.get('supporting_doc_ids') or []))); gold_servers=set().union(*(doc_servers.get(doc_id,set()) for doc_id in gold_docs))
    minimum,_=minimum_cover(gold_docs,doc_servers)
    return {'public':public,'tags':tags,'candidates':candidates,'probes':probes,'current':current,'gold_docs':gold_docs,'gold_servers':gold_servers,'minimum_required_server_count':minimum,'capabilities':capabilities,'probe_ms':sum(probe.estimated_probe_ms for probe in probes.values())}


def choices_for_case(case: dict[str,Any],selector: ProbeAwareServerSelector,max_servers: int,doc_servers: dict[str,set[str]]) -> tuple[dict[str,list[str]],dict[str,Any]]:
    public=case['public']; probe_result=selector.select(case['candidates'],case['tags'],case['probes'],len(normalize_terms(public['question'])),max_servers)
    gold_servers=case['gold_servers']; gold_docs=case['gold_docs']; capabilities=case['capabilities']; level1=set(public['level1_tags'])
    choices={
        'current_selector':case['current'],
        'probe_aware_selector':[server.server_id for server in probe_result.servers],
        'single_server_oracle':greedy_gold_order(gold_servers,gold_docs,doc_servers)[:1],
        'all_domain_servers':sorted(sid for sid,cap in capabilities.items() if level1&set(cap['level1_tags'])),
        'gold_server_oracle':greedy_gold_order(gold_servers,gold_docs,doc_servers),
    }
    return choices,{'explanation':probe_result.explanation,'score':probe_result.score,'features':dict(probe_result.features)}


def evaluate_cases(cases: list[dict[str,Any]],config: ProbeSelectorConfig,max_servers: int,doc_servers: dict[str,set[str]]) -> tuple[dict[str,Any],list[dict[str,Any]]]:
    selector=ProbeAwareServerSelector(config,max_servers); rows=[]
    for case in cases:
        started=time.perf_counter(); choices,probe_detail=choices_for_case(case,selector,max_servers,doc_servers); selection_ms=(time.perf_counter()-started)*1000
        methods={name:method_evaluation(selected,case['gold_docs'],doc_servers,case['gold_servers']) for name,selected in choices.items()}
        rows.append({'query_id':case['public']['query_id'],'source_dataset':case['public']['source_dataset'],'query_type':case['public']['query_type'],'minimum_required_server_count':case['minimum_required_server_count'],'probe_ms':case['probe_ms'],'selection_ms':selection_ms,'probe_detail':probe_detail,'methods':methods})
    return aggregate_rows(rows),rows


def aggregate_rows(rows: list[dict[str,Any]]) -> dict[str,Any]:
    output={'query_count':len(rows),'average_probe_ms':sum(row['probe_ms'] for row in rows)/len(rows) if rows else 0.0,'average_probe_selector_ms':sum(row['selection_ms'] for row in rows)/len(rows) if rows else 0.0,'methods':{}}
    for method in METHODS:
        items=[row['methods'][method] for row in rows]; selected=sum(item['selected_server_count'] for item in items); unnecessary=sum(len(item['unnecessary_servers']) for item in items); multi=[item for row,item in zip(rows,items) if row['minimum_required_server_count'] is not None and row['minimum_required_server_count']>=2]; second=[item['second_server_marginal_supporting_document_gain'] for item in items if item['second_server_marginal_supporting_document_gain'] is not None]
        output['methods'][method]={'supporting_document_recall':sum(item['supporting_document_recall'] for item in items)/len(items) if items else 0.0,'gold_server_recall':sum(item['gold_server_recall'] for item in items)/len(items) if items else 0.0,'average_selected_server_count':selected/len(items) if items else 0.0,'multi_server_required_full_recall':sum(item['supporting_document_recall']==1.0 for item in multi)/len(multi) if multi else 0.0,'unnecessary_server_rate':unnecessary/selected if selected else 0.0,'second_server_positive_gold_gain_rate':sum(gain>0 for gain in second)/len(second) if second else 0.0}
    return output


def group_rows(rows: list[dict[str,Any]],field: str) -> dict[str,Any]:
    groups=defaultdict(list)
    for row in rows: groups[str(row[field])].append(row)
    return {key:aggregate_rows(value) for key,value in sorted(groups.items())}


def objective(metrics: dict[str,Any],config: ProbeSelectorConfig) -> float:
    probe=metrics['methods']['probe_aware_selector']
    return config.objective_recall_weight*probe['supporting_document_recall']-config.objective_unnecessary_weight*probe['unnecessary_server_rate']-config.objective_server_count_weight*probe['average_selected_server_count']


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-file',required=True); parser.add_argument('--partition-stats',required=True); parser.add_argument('--registry',required=True); parser.add_argument('--edge-dir',required=True); parser.add_argument('--config',default='configs/probe_selector.yaml'); parser.add_argument('--output',required=True); parser.add_argument('--max-servers',type=int,default=4)
    args=parser.parse_args(); config=ProbeSelectorConfig.from_yaml(args.config); registry,capabilities=load_registry(Path(args.registry)); edge_servers=load_edge_servers(Path(args.edge_dir))
    doc_servers,_=load_partition_indexes(Path(args.edge_dir),Path(args.partition_stats)); baseline=ResourceAwareServerSelector(args.max_servers)
    queries=[query for query in records(Path(args.query_file)) if query.get('supporting_doc_ids')]
    cases=[prepare_case(query,registry,capabilities,edge_servers,doc_servers,baseline,config,args.max_servers) for query in queries]
    calibration=[case for case in cases if split_name(case['public']['query_id'])=='calibration']; evaluation=[case for case in cases if split_name(case['public']['query_id'])=='evaluation']
    pareto=[]; best=None
    for threshold,server_penalty,overlap_penalty in product(config.calibration_marginal_gain_thresholds,config.calibration_server_count_penalties,config.calibration_overlap_penalties):
        trial=config.tuned(marginal_gain_threshold=threshold,server_count_penalty=server_penalty,overlap_penalty=overlap_penalty); metrics,_=evaluate_cases(calibration,trial,args.max_servers,doc_servers); score=objective(metrics,config); point={'marginal_gain_threshold':threshold,'server_count_penalty':server_penalty,'overlap_penalty':overlap_penalty,'objective':score,**metrics['methods']['probe_aware_selector']}; pareto.append(point)
        if best is None or (score,-point['average_selected_server_count'])>(best[0],-best[1]['average_selected_server_count']): best=(score,point,trial)
    assert best is not None
    evaluation_metrics,rows=evaluate_cases(evaluation,best[2],args.max_servers,doc_servers)
    report={'schema_version':1,'privacy_constraint':'gold labels are attached only after baseline/probe selection','split':{'method':'sha256(query_id) modulo 100','calibration_count':len(calibration),'evaluation_count':len(evaluation)},'selected_parameters':{key:best[1][key] for key in ('marginal_gain_threshold','server_count_penalty','overlap_penalty')},'calibration_pareto':pareto,'evaluation':evaluation_metrics,'groups':{'source_dataset':group_rows(rows,'source_dataset'),'query_type':group_rows(rows,'query_type')},'queries':rows}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps({'output':str(output),'split':report['split'],'selected_parameters':report['selected_parameters'],'evaluation':evaluation_metrics},ensure_ascii=False,indent=2)); return 0


if __name__=='__main__': raise SystemExit(main())
