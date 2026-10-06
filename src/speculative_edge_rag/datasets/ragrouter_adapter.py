"""RAGRouter-Bench parsing and deterministic, model-free chunking."""
from __future__ import annotations
import hashlib, json, re, warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

def global_document_id(source_dataset: str, original_doc_id: object) -> str:
    """Return the stable document identity used across offline artifacts."""
    return f'{source_dataset}:{original_doc_id}'

@dataclass
class DocumentRecord:
    doc_id: str; title: str; text: str; source_dataset: str
    level1_tags: list[str] = field(default_factory=list); level2_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def level1_tag(self): return self.level1_tags[0] if self.level1_tags else ''
    def to_dict(self): return asdict(self)

@dataclass
class ChunkRecord:
    chunk_id: str; doc_id: str; chunk_index: int; text: str; domain: str; source_dataset: str
    level1_tags: list[str]; level2_tags: list[str]
    def to_dict(self): return asdict(self)

@dataclass
class QueryRecord:
    query_id: str; question: str; answer: Any; query_type: str; source_dataset: str
    level1_tags: list[str] = field(default_factory=list); level2_tags: list[str] = field(default_factory=list)
    supporting_facts: Any = field(default_factory=list); supporting_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    @property
    def text(self): return self.question
    def to_dict(self): return asdict(self)

class BoundaryChunker:
    def __init__(self, chunk_size=1200, chunk_overlap=150):
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size: raise ValueError('invalid chunk configuration')
        self.chunk_size, self.chunk_overlap = chunk_size, chunk_overlap
    def split(self, text):
        text = (text or '').strip()
        if not text: return []
        if len(text) <= self.chunk_size: return [text]
        ends = {m.end() for m in re.finditer(r'(?:[.!?。！？][\"\'”’）)]?\s+|\n\s*\n)', text)}
        out=[]; start=0
        while start < len(text):
            hard=min(len(text), start+self.chunk_size)
            choices=[x for x in ends if start+self.chunk_size//2 <= x <= hard]
            end=len(text) if hard==len(text) else max(choices, default=hard)
            part=text[start:end].strip()
            if part: out.append(part)
            if end>=len(text): break
            start=max(start+1,end-self.chunk_overlap)
            while start<end and text[start].isspace(): start+=1
        return out
    def chunk_document(self, doc, tag_assigner=None):
        result=[]
        for i,text in enumerate(self.split(doc.text)):
            key='\0'.join((doc.doc_id,str(i),text))
            cid='chunk-'+hashlib.sha256(key.encode()).hexdigest()[:24]
            level2 = (tag_assigner.assign_document_tags(text, doc.level1_tag)
                      if tag_assigner is not None else list(doc.level2_tags))
            result.append(ChunkRecord(cid,doc.doc_id,i,text,doc.level1_tag,doc.source_dataset,list(doc.level1_tags),list(level2)))
        return result

class TagAssigner:
    TAXONOMY={'encyclopedic':['person','organization','location','event','history','science','culture','summary'],'literature':['character','event','relationship','motivation','theme','setting','plot','summary'],'legal':['statute','case','right','obligation','liability','procedure','penalty','remedy','summary'],'medical':['disease','symptom','risk','diagnosis','test','treatment','drug','prognosis','summary']}
    KEYWORDS={k: rf'\b({"|".join(v)})\b' for k,v in {'person':['who','person','born','author','president','character'],'organization':['company','organization','agency','institution'],'location':['where','city','country','located','place'],'event':['event','war','battle','when','happened'],'history':['history','historical','century','ancient'],'science':['science','study','research','theory','species'],'culture':['culture','art','music','tradition'],'character':['character','protagonist','hero','he','she'],'relationship':['relation','married','friend','family','brother','sister'],'motivation':['why','motivat','reason','purpose'],'theme':['theme','meaning','message'],'setting':['setting','takes place','scene'],'plot':['plot','story','chapter','ending'],'statute':['statute','law','act','code','regulation'],'case':['case','court','ruling','judgment'],'right':['right','freedom','entitle'],'obligation':['obligation','duty','must','required'],'liability':['liability','liable','damages'],'procedure':['procedure','filing','process'],'penalty':['penalty','fine','sentence','punish'],'remedy':['remedy','relief','compensation'],'disease':['disease','cancer','diabetes','infection'],'symptom':['symptom','pain','fever','cough'],'risk':['risk','cause','factor','prevention'],'diagnosis':['diagnos','condition'],'test':['test','scan','exam','laboratory'],'treatment':['treatment','therapy','surgery'],'drug':['drug','medication','medicine'],'prognosis':['prognosis','outcome','survival','recover']}.items()}
    def __init__(self,taxonomy=None): self.taxonomy=taxonomy or self.TAXONOMY
    def assign_document_tags(self,text,level1):
        tags=[t for t in self.taxonomy.get(level1,[]) if re.search(self.KEYWORDS.get(t,r'(?!)'),text.lower())]; return tags or ['summary']
    def assign_query_tags(self,text,level1=None):
        levels=[level1] if level1 else list(self.taxonomy); tags=[t for l in levels for t in self.taxonomy[l] if re.search(self.KEYWORDS.get(t,r'(?!)'),text.lower())]; return list(dict.fromkeys(tags)) or ['summary']

class RAGRouterDatasetAdapter:
    MAX_JSONL_RECORD_LINES = 32
    MAX_JSONL_RECORD_CHARS = 2_000_000
    DOMAIN_MAP={'musique':'encyclopedic','quality':'literature','ultraDomain_legal':'legal','graphragBench_medical':'medical','encyclopedic':'encyclopedic','literature':'literature','legal':'legal','medical':'medical'}
    def __init__(self,tag_assigner=None):
        self.tagger=tag_assigner or TagAssigner(); self.parse_failures={}; self.parse_errors=[]; self.recovered_records={}
    def to_query(self,record): return self.parse_queries([record],str(record.get('source_dataset','encyclopedic')))[0]
    def _items(self,obj,keys):
        if isinstance(obj,list): return obj
        if isinstance(obj,dict):
            for key in (*keys,'data','records'):
                if isinstance(obj.get(key),list): return obj[key]
            return [obj]
        return []
    @staticmethod
    def _inside_json_string(text):
        inside=False; escaped=False
        for char in text:
            if escaped: escaped=False
            elif char == '\\': escaped=True
            elif char == '"': inside=not inside
        return inside
    def _report_parse_error(self,path,domain,kind,start_line,end_line,error):
        self.parse_failures.setdefault(domain,{'documents':0,'queries':0})[kind]+=1
        detail={'path':str(path),'kind':kind,'start_line':start_line,'end_line':end_line,'error':str(error)}
        self.parse_errors.append(detail)
        warnings.warn(f"unrecoverable JSON record in {path}:{start_line}-{end_line}: {error}",RuntimeWarning,stacklevel=2)
    def _read_jsonl(self,path,domain,kind,text):
        rows=[]; buffer=''; start_line=0; buffered_lines=0; last_error=None
        for line_number,line in enumerate(text.splitlines(),1):
            if not line.strip() and not buffer: continue
            if not buffer:
                try: rows.append(json.loads(line)); continue
                except json.JSONDecodeError as error:
                    buffer=line; start_line=line_number; buffered_lines=1; last_error=error; continue
            separator='\\n' if self._inside_json_string(buffer) else '\n'
            candidate=buffer+separator+line
            try:
                rows.append(json.loads(candidate))
                self.recovered_records.setdefault(domain,{'documents':0,'queries':0})[kind]+=1
                buffer=''; buffered_lines=0; last_error=None
                continue
            except json.JSONDecodeError as error:
                last_error=error
            try:
                standalone=json.loads(line)
            except json.JSONDecodeError:
                standalone=None
            else:
                self._report_parse_error(path,domain,kind,start_line,line_number-1,last_error)
                rows.append(standalone); buffer=''; buffered_lines=0; last_error=None
                continue
            buffer=candidate; buffered_lines+=1
            if buffered_lines >= self.MAX_JSONL_RECORD_LINES or len(buffer) >= self.MAX_JSONL_RECORD_CHARS:
                self._report_parse_error(path,domain,kind,start_line,line_number,last_error)
                buffer=''; buffered_lines=0; last_error=None
        if buffer:
            self._report_parse_error(path,domain,kind,start_line,start_line+buffered_lines-1,last_error)
        return rows
    def _read(self,path,domain,kind):
        text=path.read_text(encoding='utf-8')
        try: return json.loads(text)
        except json.JSONDecodeError: return self._read_jsonl(path,domain,kind,text)
    def parse_documents(self,obj,source_dataset,limit=None):
        level1=self.DOMAIN_MAP.get(source_dataset,source_dataset); out=[]
        for i,d in enumerate(self._items(obj,('documents',))[:limit]):
            if not isinstance(d,dict): continue
            raw=str(d.get('id',d.get('doc_id',f'index-{i}'))); title=str(d.get('title',d.get('name',raw))); text=str(d.get('text',d.get('contents',d.get('context',d.get('document','')))))
            doc_id=global_document_id(source_dataset,raw)
            out.append(DocumentRecord(doc_id,title,text,source_dataset,[level1],self.tagger.assign_document_tags(title+' '+text,level1),{'original_doc_id':raw,'source_doc_id':raw}))
        return out
    def parse_queries(self,obj,source_dataset,limit=None):
        level1=self.DOMAIN_MAP.get(source_dataset,source_dataset); out=[]
        for i,q in enumerate(self._items(obj,('questions',))[:limit]):
            if not isinstance(q,dict): continue
            question=str(q.get('question',q.get('query',q.get('text','')))); facts=q.get('supporting_facts',[]) or []
            if 'supporting_doc_ids' in q:
                ids=q.get('supporting_doc_ids') or []; supporting_source='explicit'
            else:
                ids=[fact.get('doc_id') for fact in facts if isinstance(fact,dict) and fact.get('doc_id') is not None]
                supporting_source='derived_from_supporting_facts' if ids else 'unavailable'
            supporting_doc_ids=list(dict.fromkeys(global_document_id(source_dataset,raw_id) for raw_id in ids))
            out.append(QueryRecord(str(q.get('question_id',q.get('query_id',q.get('id',i)))),question,q.get('answer',''),str(q.get('query_type',q.get('type',''))),source_dataset,[level1],self.tagger.assign_query_tags(question,level1),facts,supporting_doc_ids,{'supporting_doc_ids_source':supporting_source}))
        return out
    def load(self,root,limit_per_dataset=None):
        root=Path(root); docs=[]; queries=[]; self.parse_failures={}; self.parse_errors=[]; self.recovered_records={}
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name.lower()=='corpus.json':
                ds=next((x for x in p.parts if x in self.DOMAIN_MAP),p.parent.name); docs+=self.parse_documents(self._read(p,ds,'documents'),ds,limit_per_dataset)
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.name.lower()=='question.json':
                ds=next((x for x in p.parts if x in self.DOMAIN_MAP),p.parent.name); queries+=self.parse_queries(self._read(p,ds,'queries'),ds,limit_per_dataset)
        return docs,queries
RAGRouterBenchAdapter=RAGRouterDatasetAdapter
DatasetQuery=QueryRecord
