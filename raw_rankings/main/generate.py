"""Run one paired synthetic-to-human benchmark cell on a public corpus."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


P = json.loads(Path(__file__).with_name("corpus_sweep_protocol.json").read_text())


def representation(fit,other):
    word = TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=18000,sublinear_tf=True,dtype=np.float32)
    char = TfidfVectorizer(analyzer="char",ngram_range=(3,5),min_df=2,max_features=24000,
                           sublinear_tf=True,dtype=np.float32)
    a,b = word.fit_transform(fit),word.transform(other)
    c,d = char.fit_transform(fit),char.transform(other)
    svd = TruncatedSVD(min(64,a.shape[1]-1,a.shape[0]-1),random_state=26112499).fit(a)
    return (a,b),(c,d),(normalize(svd.transform(a)),normalize(svd.transform(b)))


def text_embed(model, values):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model,local_files_only=True)
    tokenizer.padding_side="left"
    encoder = AutoModel.from_pretrained(model,local_files_only=True,dtype=torch.bfloat16,
                                        attn_implementation="sdpa").cuda().eval()
    rows = []
    with torch.inference_mode():
        for start in range(0,len(values),128):
            tokens=tokenizer(values[start:start+128],padding=True,truncation=True,max_length=64,
                             return_tensors="pt").to("cuda")
            hidden=encoder(**tokens,use_cache=False).last_hidden_state
            rows.append(hidden[:,-1].float().cpu().numpy())
    return normalize(np.concatenate(rows))


def images_embed(values):
    import torch
    from torchvision.models import resnet18
    from vision import WEIGHTS, embedding
    x=np.asarray(values,dtype=np.uint8).reshape(len(values),-1,32,32) if np.asarray(values[0]).size!=784 else None
    if x is None:
        x=np.repeat(np.asarray(values,dtype=np.uint8).reshape(-1,1,28,28),3,axis=1)
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True))
    model.fc=torch.nn.Identity()
    model=model.to(device="cuda",memory_format=torch.channels_last).eval()
    return embedding(model,x)


def heads(a,b,y):
    lr=LogisticRegression(C=1,max_iter=350).fit(a,y)
    return lr.predict_proba(b)


def geometry(a,b,y):
    a,b=normalize(a),normalize(b)
    classes=np.unique(y)
    centers=normalize(np.stack([a[y==c].mean(0) for c in classes]))
    sim=b@centers.T
    ids=NearestNeighbors(n_neighbors=10,algorithm="brute").fit(a).kneighbors(b,return_distance=False)
    return [sim.max(1,keepdims=True)-sim,(y[ids,None]!=classes[None,None,:]).sum(1)]


def make_scores(fit,rest,y,name,model):
    x,z=[r[2] for r in fit],[r[2] for r in rest]
    text=isinstance(x[0],str)
    if text:
        word,char,latent=representation(x,z)
        clean_proba=heads(*word,y)
        char_proba=heads(*char,y)
        core=["word-head","char-head","margin","10NN"]
        mats=[-np.log(clean_proba),-np.log(char_proba),*geometry(*latent,y)]
        alt=char
        emb=text_embed(model,x+z)
    else:
        a,b=np.asarray(x).reshape(len(x),-1),np.asarray(z).reshape(len(z),-1)
        if name=="AudioSet":
            emb=normalize(np.concatenate([a[:,128:],b[:,128:]]))
            a,b=a[:,:128],b[:,:128]
        else:
            emb=images_embed(x+z)
        scaler=StandardScaler().fit(a)
        a,b=scaler.transform(a),scaler.transform(b)
        clean_proba=heads(a,b,y)
        core=["numeric-head","margin","10NN"]
        mats=[-np.log(clean_proba),*geometry(a,b,y)]
        alt=normalize(a),normalize(b)
    a,b=emb[:len(fit)],emb[len(fit):]
    proba=heads(a,b,y)
    mats.extend([-np.log(proba),*geometry(a,b,y)])
    aux=RidgeClassifier(alpha=1).fit(alt[0],y)
    d=aux.decision_function(alt[1])
    if d.ndim==1:
        d=np.stack([-d,d],axis=1)
    return core,["frozen-head","frozen-margin","frozen-10NN"],np.stack(mats),proba,softmax(d,axis=1)


def orders(mats,proba,claim,tie):
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    rank=[]
    for v in mats[:,np.arange(len(claim)),claim]:
        rank.append(tie[np.argsort(-v[tie],kind="stable")])
    issues=find_label_issues(claim,proba,filter_by="prune_by_noise_rate",
                            return_indices_ranked_by="normalized_margin",n_jobs=1)
    rest=tie[~np.isin(tie,issues)]
    quality=get_label_quality_scores(claim,proba,method="normalized_margin")
    rank.append(np.r_[issues,rest[np.argsort(quality[rest],kind="stable")]])
    return np.stack(rank)



import gzip

def run(name, model, out):
    with gzip.open(Path(__file__).parent/'inputs'/f'{name}.jsonl.gz', 'rt', encoding='utf8') as stream:
        rows=[json.loads(line) for line in stream]
    f,h,t=[[r for r in rows if r['role']==i] for i in range(3)]
    classes=np.unique([r['reference'] for r in f])
    y,hy,ty,claim=[np.searchsorted(classes,[r[field] for r in group]) for group,field in ((f,'reference'),(h,'reference'),(t,'reference'),(t,'claim'))]
    def rec(r): return (None,None,r['text'])
    base,added,mats,proba,aux=make_scores(list(map(rec,f)),list(map(rec,h+t)),y,name,model)
    names=base+added+['Confident-Learning']; size=len(h); e=ty!=claim
    target=orders(mats[:,size:],proba[size:],claim,np.argsort([r['tie'] for r in t]))
    aux=aux[:size].copy();aux[np.arange(size),hy]=-1
    dst,difficulty=aux.argmax(1),aux.max(1)
    rows=[]
    for seed in P['seeds']:
        rng=np.random.default_rng(seed)
        ix=np.concatenate([rng.permutation(np.flatnonzero(hy==c))[:max(1,round(.8*(hy==c).sum()))] for c in np.unique(hy)])
        k=round(P['noise_rate']*len(ix));tie=rng.permutation(len(ix))
        masks=[rng.permutation(len(ix))[:k],tie[np.argsort(-difficulty[ix][tie],kind='stable')[:k]]]
        source_y=hy[ix];history=[]
        for mask in masks:
            c=source_y.copy();c[mask]=dst[ix[mask]]
            history.append((c!=source_y)[orders(mats[:,ix],proba[ix],c,tie)])
        for b in P['budgets']:
            sh=np.stack([v[:,:round(len(ix)*b)].sum(1) for v in history]);th=e[target[:,:round(len(e)*b)]].sum(1)
            for menu,n in (('core',len(base)),('expanded',len(names))):
                choices=sh[:,:n].argmax(1)
                rows.append(dict(seed=seed,budget=b,menu=menu,source_n=len(ix),noise=k,mask_overlap=int(np.intersect1d(*masks).size),random=int(choices[0]),difficulty=int(choices[1]),oracle=int(th[:n].argmax()),random_hits=int(th[choices[0]]),difficulty_hits=int(th[choices[1]]),oracle_hits=int(th[:n].max()),random_source=sh[0,:n].tolist(),difficulty_source=sh[1,:n].tolist()))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'paired.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(out/'target.npz',errors=e,rank=target,names=np.array(names),ids=np.arange(len(t)),group=np.array([r['group'] for r in t]))
    print(name,len(f),len(h),len(t),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('datasets',nargs='+');p.add_argument('--model',type=Path);p.add_argument('--output',type=Path,default=Path(__file__).parent/'generated');a=p.parse_args()
    import torch
    torch.set_num_threads(3);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    for name in a.datasets: run(name,a.model,a.output/name)
