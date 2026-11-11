import argparse,csv,gzip,json
from pathlib import Path
from acoustic import embed

p=argparse.ArgumentParser()
p.add_argument('release',type=Path)
p.add_argument('--output',type=Path,default=Path(__file__).parent/'inputs/AudioSet.jsonl.gz')
a=p.parse_args()
with (a.release/'weak_eval.csv').open(encoding='utf8',newline='') as f:
    for _ in range(3):next(f)
    weak={(v,round(float(start)*1000)):(float(end),set(labels.split(','))) for v,start,end,labels in csv.reader(f,skipinitialspace=True)}
ontology=json.loads((a.release/'ontology.json').read_text(encoding='utf8'))
children={r['id']:r['child_ids'] for r in ontology}
todo=[next(r['id'] for r in ontology if r['name']=='Speech')];family=set()
while todo:
    tag=todo.pop()
    if tag not in family:
        family.add(tag);todo.extend(children.get(tag,[]))
coverage={}
with (a.release/'strong.tsv').open(encoding='utf8',newline='') as f:
    for r in csv.DictReader(f,delimiter='\t'):
        video,ms=r['segment_id'].rsplit('_',1);key=(video,int(ms))
        event=r['label'] in family and float(r['end_time_seconds'])-float(r['start_time_seconds'])>=.5
        coverage[key]=int(coverage.get(key,False) or event)
features=embed(a.release/'features/eval',weak)
fixed=Path(__file__).parent/'inputs/AudioSet.jsonl.gz'
with gzip.open(fixed,'rt',encoding='utf8') as f:
    rows=[json.loads(line) for line in f]
for row in rows:
    video,ms=row['item'].rsplit(':',1);key=(video,int(ms))
    row['text']=features[key].tolist()
    row['reference']=coverage[key]
    row['claim']=int(bool(weak[key][1]&family))
with gzip.open(a.output,'wt',encoding='utf8') as f:
    for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print('AudioSet',len(rows))
