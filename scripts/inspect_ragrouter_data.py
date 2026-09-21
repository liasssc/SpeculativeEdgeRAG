#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from collections import Counter

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--processed-dir',default='data/processed'); ap.add_argument('--partition-dir',default='data/server_partitions'); a=ap.parse_args(); p=Path(a.processed_dir)
 docs=[]; qs=[]
 for f in p.glob('documents*.jsonl'):
  docs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
 for f in p.glob('queries*.jsonl'):
  qs += [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
 print('documents:',len(docs)); print('queries:',len(qs)); print('domains:',dict(Counter(d.get('level1_tag') for d in docs))); print('tags:',dict(Counter(t for d in docs for t in d.get('level2_tags',[]))))
 ids={str(d.get('doc_id')) for d in docs}; mapped=total=multi=0
 for q in qs:
  s=q.get('supporting_doc_ids',[]) or []; total+=len(s); mapped+=sum(str(x) in ids for x in s); multi += len(set(q.get('level2_tags',[])))>=2
 print('supporting_facts mapped:',f'{mapped}/{total}' if total else '0/0'); print('queries with >=2 tags:',multi)
 server_docs={}
 for f in sorted(Path(a.partition_dir).glob('*/documents.jsonl')):
  ds=[json.loads(x) for x in f.read_text().splitlines() if x.strip()]; server_docs[f.parent.name]={str(d.get('doc_id')) for d in ds}; print(f.parent.name, 'documents=',len(ds), 'tags=',sorted({t for d in ds for t in d.get('level2_tags',[])}), 'tokens=',sum(len((d.get('text') or '').split()) for d in ds))
 covered=0
 for q in qs:
  s={str(x) for x in (q.get('supporting_doc_ids',[]) or [])}
  if sum(bool(s & ids) for ids in server_docs.values()) >= 2: covered += 1
 print('queries covered by multiple edge servers:', covered)
if __name__=='__main__': main()
