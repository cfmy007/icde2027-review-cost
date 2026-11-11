"""Five-way label intervention on the same selected CIFAR-10N records."""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prepare_features import CFG, HERE, import_script, load_cifar


def norm(z):
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-8)


def run(base, device):
    old = import_script(base/'scripts/107_endtoend_more.py')
    images, clean, human = load_cifar(base, 'c10')
    with np.load(HERE/'inputs/c10_features.npz') as pack:
        idx = pack['rest']
        feats = norm(pack['cnn'])
    nref = CFG['cifar_ref_n']
    ref_y = clean[idx[:nref]]
    pool = idx[nref:]
    centers = norm(np.stack([feats[:nref][ref_y == c].mean(axis=0) for c in range(10)]))
    sim = feats[nref:] @ centers.T
    clean, obs = clean[pool], human['worse_label'][pool]
    err = clean != obs
    top = sim.argmax(axis=1)
    maximum = sim.max(axis=1)
    score = maximum - sim[np.arange(len(pool)), obs]
    ties = np.random.default_rng(74109).permutation(len(pool))
    sel = ties[np.argsort(-score[ties], kind='stable')][:len(pool)//10]
    fixed = sel[err[sel] & (top[sel] == clean[sel])]
    damaged = sel[~err[sel] & (top[sel] != clean[sel])]
    unfixed = sel[err[sel] & (top[sel] != clean[sel])]
    assigned = {k: obs.copy() for k in ('none', 'auto', 'fix_only', 'damage_only', 'review')}
    assigned['auto'][sel] = top[sel]
    assigned['fix_only'][fixed] = clean[fixed]
    assigned['damage_only'][damaged] = top[damaged]
    assigned['review'][sel] = clean[sel]
    second = np.partition(sim, -2, axis=1)[:, -2]
    groups = {'all_errors': np.flatnonzero(err), 'all_clean': np.flatnonzero(~err), 'fixed': fixed,
              'damaged': damaged, 'selected_unfixed': unfixed}
    geometry = {}
    for tag, subset in groups.items():
        geometry[tag] = dict(n=len(subset), nearest_gap=float(np.mean((maximum-second)[subset])),
                             clean_similarity=float(np.mean(sim[subset, clean[subset]])),
                             wrong_margin=float(np.mean(score[subset])),
                             per_class=np.bincount(clean[subset], minlength=10).tolist())
    dataset = pickle.loads((base/'data/raw/cifar10/cifar-10-batches-py/test_batch').read_bytes(), encoding='bytes')
    xt = torch.from_numpy(dataset[b'data'].reshape(-1,3,32,32)).to(device).float().div_(255)
    yt = torch.as_tensor(dataset[b'labels'], device=device)
    train_x = torch.from_numpy(images[pool]).to(device).float().div_(255)
    train_y = {k: torch.as_tensor(v, device=device) for k, v in assigned.items()}
    output = dict(n=len(pool), k=len(sel), n_errors=int(err.sum()), geometry=geometry,
                  net_labels={tag: int((v == clean).sum() - (~err).sum()) for tag, v in assigned.items()}, runs=[])
    path = HERE/'results/downstream.json'
    for seed in CFG['seeds']:
        row = dict(seed=seed, accuracy={})
        for tag, labels in train_y.items():
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            model = old.PlainCNN().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(12):
                for ix in torch.as_tensor(rng.permutation(len(pool)), device=device).split(128):
                    xb = train_x[ix]
                    if rng.random() < .5:
                        xb = xb.flip([3])
                    xb = F.pad(xb, (4, 4, 4, 4), mode='reflect')
                    ox, oy = rng.integers(0, 9, size=2)
                    xb = xb[:, :, oy:oy+32, ox:ox+32].contiguous()
                    opt.zero_grad()
                    F.cross_entropy(model(xb), labels[ix]).backward()
                    opt.step()
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(batch).argmax(1) for batch in xt.split(512)])
                row['accuracy'][tag] = int((pred == yt).sum()) / len(yt)
        output['runs'].append(row)
        path.write_text(json.dumps(output, indent=2))
        print(seed, row['accuracy'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    run(a.base.resolve(), a.device)
