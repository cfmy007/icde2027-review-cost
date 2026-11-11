import argparse,gzip,json
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('datasets',nargs='+')
p.add_argument('--output',type=Path,default=root/'inputs')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
for name in a.datasets:
    with gzip.open(root/'source'/f'{name}.jsonl.gz','rt',encoding='utf8') as f:
        source=[json.loads(line) for line in f]
    with gzip.open(root/'inputs'/f'{name}.jsonl.gz','rt',encoding='utf8') as f:
        rows=[json.loads(line) for line in f]
    for row in rows:
        raw=source[row['source_index']]
        row['text']=raw.get('text') or raw['votes'][0][1]
        row['reference']=raw['reference']
        row['claim']=raw['votes'][0][0]
    with gzip.open(a.output/f'{name}.jsonl.gz','wt',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    print(name,len(rows),flush=True)
