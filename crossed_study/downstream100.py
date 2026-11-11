"""Paired 100-seed downstream comparison for two historical operating points."""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from downstream import norm
from prepare_features import CFG, HERE, fit, import_script, infer, load_cifar

CONFIG = json.loads((HERE/'downstream100_protocol.json').read_text())
SEEDS = tuple(range(CONFIG['seed_start'], CONFIG['seed_start']+CONFIG['seed_count']))
SETUPS = ('fixed10', 'legacy_calibrated')


def commit(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    temp.replace(path)


def legacy_detector(base, x, obs, path, dev):
    if path.exists():
        with np.load(path) as data:
            return data['margin'], data['proposed']
    orig = import_script(base/'scripts/107_endtoend_more.py')
    cmod = orig.c100n
    splits = np.split(np.random.default_rng(0).permutation(len(obs)),2)
    margin = np.zeros(len(obs),dtype=np.float32)
    proposed = np.zeros(len(obs),dtype=np.int64)
    for seed,(tr,te) in enumerate(((splits[1],splits[0]),(splits[0],splits[1]))):
        torch.manual_seed(seed)
        fit_x = x[tr].astype(np.float32)/255.
        test_x = x[te].astype(np.float32)/255.
        model,_ = fit(cmod.CNN(10),fit_x,obs[tr],8,seed,dev,True)
        zr = norm(infer(model,fit_x,dev)[0])
        zt = norm(infer(model,test_x,dev)[0])
        dis = 1. - zt @ orig.centroids(zr,obs[tr]).T
        margin[te] = dis[np.arange(len(te)),obs[te]]-dis.min(1)
        proposed[te] = dis.argmin(1)
        print(f'Legacy detector fold {seed} frozen',flush=True)
    np.savez_compressed(path,margin=margin,proposed=proposed)
    return margin,proposed


def fixed_pool(clean, human, cache):
    with np.load(cache/'c10_features.npz') as pack:
        idx = pack['rest']
        assert np.array_equal(clean,pack['y'])
        feats = norm(pack['cnn'])
    nref = CFG['cifar_ref_n']
    ref_y = clean[idx[:nref]]
    centers = norm(np.stack([feats[:nref][ref_y==c].mean(axis=0) for c in range(10)]))
    pool = idx[nref:]
    sim = feats[nref:] @ centers.T
    obs = human[pool]
    score = sim.max(1)-sim[np.arange(len(pool)),obs]
    ties = np.random.default_rng(CONFIG['fixed_tie_seed']).permutation(len(pool))
    order = ties[np.argsort(-score[ties],kind='stable')]
    sel = order[:round(CONFIG['fixed_action_fraction']*len(pool))]
    return pool, sim.argmax(1),sel


def target_counts(obs, clean, suggest, selected):
    auto = obs.copy()
    auto[selected] = suggest[selected]
    fixed = int(((obs!=clean)&(auto==clean)).sum())
    damaged = int(((obs==clean)&(auto!=clean)).sum())
    assert int((auto==clean).sum())-int((obs==clean).sum())==fixed-damaged
    return auto, fixed, damaged


def schedule(n, seed, dev):
    rng = np.random.default_rng(seed)
    axis = np.arange(32,dtype=np.int64)
    b = CONFIG['batch']
    plans = []
    for _ in range(CONFIG['epochs']):
        perm = rng.permutation(n)
        offsets = []
        for _ in range(0,n,b):
            flip = rng.random()<.5
            ox,oy = rng.integers(0,9,size=2)
            x_coord = 39-ox-axis if flip else ox+axis
            offsets.append(np.stack([oy+axis,x_coord]))
        # Store the full and last partial batches separately.
        rows = torch.as_tensor(perm,device=dev)
        coords = torch.as_tensor(np.stack(offsets),device=dev)
        plans.append((rows,coords))
    return plans


class Step:
    def __init__(self, model, optimizer, padded, labels, idx, coord, graph):
        self.model,self.opt,self.padded,self.labels = model,optimizer,padded,labels
        self.idx,self.coord = idx.clone(),coord.clone()
        self.graph = None
        if graph:
            # Initialize optimizer state using the real first update in run().
            self.graph = torch.cuda.CUDAGraph()
            self.is_captured = False

    def update(self):
        coord = self.coord
        nhwc = self.padded[self.idx[:,None,None],coord[0,None,:,None],coord[1,None,None,:]]
        self.opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False,enabled=nhwc.is_cuda):
            out = self.model(nhwc.permute(0,3,1,2))
        F.cross_entropy(out.float(),self.labels[self.idx]).backward()
        self.opt.step()

    def run(self, idx, coord):
        self.idx.copy_(idx)
        self.coord.copy_(coord)
        if self.graph is None:
            self.update()
        elif not self.is_captured:
            # Capture after a real warmup update in train_pair_arm().
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph):
                self.update()
            self.graph.replay()
            self.is_captured = True
        else:
            self.graph.replay()


def train_pair_arm(original,padded,lab,plans,seed,x_test,y_test,dev,return_hits=False):
    torch.manual_seed(seed)
    model = original.PlainCNN().to(device=dev,memory_format=torch.channels_last)
    use_graph = dev.startswith('cuda')
    opt = torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-4,
                           capturable=use_graph,fused=use_graph)
    steps = {}
    for perm, coords in plans:
        for idx, coord in zip(perm.split(CONFIG['batch']),coords.unbind()):
            k = len(idx)
            if k not in steps:
                steps[k] = Step(model,opt,padded,lab,idx,coord,use_graph)
                steps[k].update()
            else:
                steps[k].run(idx,coord)
    model.eval()
    with torch.no_grad():
        hits = [model(x).argmax(1)==y for x,y in zip(x_test.split(512),y_test.split(512))]
        if return_hits:
            return torch.cat(hits).cpu().numpy()
        return int(sum(h.sum() for h in hits))


def padded_images(images,dev):
    # Pad once. A horizontal reflection commutes with symmetric padding.
    chunks = []
    for chunk in torch.from_numpy(images).split(4096):
        x = chunk.to(dev).float().div_(255)
        chunks.append(F.pad(x,(4,4,4,4),mode='reflect').permute(0,2,3,1).contiguous())
    return torch.cat(chunks)


def run(args):
    dev = args.device
    original = import_script(args.base/'scripts/107_endtoend_more.py')
    x, clean, human = load_cifar(args.base,'c10')
    obs = human['worse_label']
    args.out.mkdir(parents=True,exist_ok=True)
    outf = args.out/'downstream100_runs.json'
    g = json.loads(outf.read_text()) if outf.exists() else {'protocol':CONFIG,'test_n':10000,'setups':{}}
    assert g['protocol']==CONFIG
    data = pickle.loads((args.base/'data/raw/cifar10/cifar-10-batches-py/test_batch').read_bytes(),encoding='bytes')
    xt = torch.from_numpy(data[b'data'].reshape(-1,3,32,32)).to(dev).float().div_(255).contiguous(memory_format=torch.channels_last)
    yt = torch.as_tensor(data[b'labels'],device=dev,dtype=torch.long)
    assert len(yt)==g['test_n']
    for setup in args.setups:
        if setup=='legacy_calibrated':
            pool = np.arange(len(obs))
            margin,suggest = legacy_detector(args.base,x,obs,args.out/'legacy_oof_100.npz',dev)
            legacy_order = np.argsort(-margin)
        else:
            pool,suggest,fixed_sel = fixed_pool(clean,obs,args.cache)
        true = clean[pool]
        lab = obs[pool]
        results = g['setups'].setdefault(setup,{'n':len(pool),'runs':[]})
        done = {r['seed'] for r in results['runs']}
        pending = [s for s in SEEDS if s not in done]
        if args.limit is not None:
            pending = pending[:args.limit]
        if not pending:
            continue
        t = time.monotonic()
        xp = padded_images(x[pool],dev)
        base_y = torch.as_tensor(lab,device=dev,dtype=torch.long)
        for seed in pending:
            if setup=='legacy_calibrated':
                audit = np.random.default_rng(seed).choice(len(pool),round(CONFIG['legacy_audit_fraction']*len(pool)),replace=False)
                frac = original.pick_budget(lab[audit]!=true[audit],margin[audit])
                selected = legacy_order[:round(frac*len(pool))]
                overlap = len(np.intersect1d(audit,selected))
            else:
                selected = fixed_sel
                audit,overlap = (),0
            auto,fixed,damaged = target_counts(lab,true,suggest,selected)
            if setup=='fixed10':
                assert fixed==1462 and damaged==54
            plans = schedule(len(pool),seed,dev)
            auto_y = torch.as_tensor(auto,device=dev,dtype=torch.long)
            t0=time.monotonic()
            none_correct = train_pair_arm(original,xp,base_y,plans,seed,xt,yt,dev)
            auto_correct = train_pair_arm(original,xp,auto_y,plans,seed,xt,yt,dev)
            results['runs'].append(dict(seed=seed,none_correct=none_correct,auto_correct=auto_correct,
                                        actions=len(selected),audit=len(audit),audit_overlap=overlap,
                                        fixed=fixed,damaged=damaged,net_fixes=fixed-damaged))
            commit(outf,g)
            print(f'{setup} {len(results["runs"]):3d}/100 seed={seed} pair={time.monotonic()-t0:.1f}s',flush=True)
        del xp
        print(f'{setup} elapsed {time.monotonic()-t:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--cache',type=Path,default=HERE/'inputs')
    parser.add_argument('--out',type=Path,default=HERE/'results')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--setups',nargs='+',choices=SETUPS,default=SETUPS)
    args=parser.parse_args()
    args.base=args.base.resolve()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    run(args)
