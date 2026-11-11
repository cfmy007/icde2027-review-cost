"""Fit only the missing frozen encoders. No evaluation results enter training."""
import argparse
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'protocol.json').read_text())


def import_script(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_cifar(root, name):
    raw = root / 'data' / 'raw'
    if name == 'c10':
        directory = raw / 'cifar10/cifar-10-batches-py'
        files, key = [directory / f'data_batch_{i}' for i in range(1, 6)], b'labels'
        labels = raw / 'cifar10n/CIFAR-10_human.pt'
        keys = ('aggre_label', 'random_label1', 'random_label2', 'random_label3', 'worse_label')
    else:
        files, key = [raw / 'cifar100/cifar-100-python/train'], b'fine_labels'
        labels = raw / 'cifar100n/CIFAR-100_human.pt'
        keys = ('noisy_label',)
    parts = [pickle.loads(p.read_bytes(), encoding='bytes') for p in files]
    images = np.concatenate([d[b'data'] for d in parts]).reshape(-1, 3, 32, 32)
    clean = np.concatenate([d[key] for d in parts])
    human = torch.load(labels, map_location='cpu', weights_only=False)
    assert np.array_equal(clean, np.asarray(human['clean_label']))
    return images, clean, {k: np.asarray(human[k]) for k in keys}


@torch.no_grad()
def infer(model, images, device):
    model.eval()
    features = []
    w = F.normalize(model.Wc, dim=1)
    for batch in torch.from_numpy(images).split(512):
        f = model.fc(model.features(batch.to(device)).flatten(1))
        features.append(f.cpu().numpy())
    feats = np.concatenate(features)
    cos = F.normalize(torch.from_numpy(feats).to(device), dim=1) @ w.T
    return feats, cos.cpu().numpy()


def fit(model, images, labels, epochs, seed, device, aug, infer_images=None):
    # Same optimizer, model and per-batch augmentation as the supplied recipes.
    model = model.to(device)
    x = torch.from_numpy(images).to(device)
    y = torch.as_tensor(labels, device=device, dtype=torch.long)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4 if aug else 0.)
    avg = None
    for ep in range(epochs):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for idx in torch.as_tensor(rng.permutation(len(y)), device=device).split(128):
            xb = x[idx]
            if aug:
                if rng.random() < .5:
                    xb = xb.flip([3])
                xb = F.pad(xb, (4, 4, 4, 4), mode='reflect')
                ox, oy = rng.integers(0, 9, size=2)
                xb = xb[:, :, oy:oy + 32, ox:ox + 32].contiguous()
            opt.zero_grad()
            loss = F.cross_entropy(model(xb, y[idx]), y[idx])
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
        print(f'fit seed={seed} epoch={ep+1}/{epochs} loss={loss_sum.item()/len(y):.4f}', flush=True)
        if infer_images is not None:
            _, cos = infer(model, infer_images, device)
            logits = 30 * cos
            prob = np.exp(logits - logits.max(1, keepdims=True))
            prob /= prob.sum(1, keepdims=True)
            avg = prob if avg is None else avg + prob
    return model, None if avg is None else avg / epochs


def chen_summary(avg, labels):
    wrong = avg.copy()
    wrong[np.arange(len(labels)), labels] = -np.inf
    return {'chen_dest': wrong.argmax(1).astype(np.int16), 'chen_conf': wrong.max(1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frozen', action='store_true', help='Reuse shipped weights and Chen destination summaries')
    parser.add_argument('--out', type=Path, default=HERE/'inputs')
    parser.add_argument('dataset', choices=['hwdb', 'c10', 'c100'])
    args = parser.parse_args()
    base, dev, name = args.base.resolve(), args.device, args.dataset
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    frozen = HERE/'inputs'
    if name == 'hwdb':
        cb = import_script(base / 'scripts/29_curated_bench.py')
        test = cb.load_cache(base / 'data/processed/hwdb11_test_32.npz')
        sub = cb.build_named_subset(test, [ch for p in cb.PAIRS for ch in p], 40, 0)
        x = (sub['X'].astype(np.float32) / 255.)[:, None]
        checkpoint = torch.load(base / 'results/curated_bench/model.pt', map_location='cpu', weights_only=False)
        assert checkpoint['names'] == sub['names']
        model = cb.CNN()
        model.load_state_dict(checkpoint['state'])
        cnn, logits = infer(model.to(dev), x, dev)
        if args.frozen:
            model = cb.CNN().to(dev)
            model.load_state_dict(torch.load(frozen/'hwdb_aux.pt', map_location='cpu'))
            temporal = dict(np.load(frozen/'hwdb_chen.npz'))
        else:
            cache = base / 'data/processed'
            xtrain, ytrain, _, _ = cb.load_train(','.join(str(cache / f'hwdb11_train{i}_100_32.npz') for i in (1, 2)), sub['names'])
            seed = CFG['hwdb_cnn_aux_seed']
            torch.manual_seed(seed)
            model, avg = fit(cb.CNN(), xtrain[:, None], ytrain, CFG['hwdb_epochs'], seed, dev, False, x)
            torch.save(model.cpu().state_dict(), out / 'hwdb_aux.pt')
            temporal = chen_summary(avg, sub['y'])
        aux, aux_logits = infer(model.to(dev), x, dev)
        np.savez_compressed(out / 'hwdb_features.npz', cnn=cnn, aux=aux, logits=logits, aux_logits=aux_logits,
                            y=sub['y'], writer=sub['writer'], names=sub['names'], pair_map=checkpoint['pair_map'], **temporal)
        return
    cmod = import_script(base / 'scripts/71_cifar100n.py')
    x, y, _ = load_cifar(base, name)
    perm = np.random.default_rng(CFG['fit_seed']).permutation(len(y))
    fit_idx, rest = np.split(perm, [CFG['cifar_fit_n']])
    xrest = x[rest].astype(np.float32) / 255.
    saved = {'fit_idx': fit_idx, 'rest': rest, 'y': y}
    if not args.frozen:
        xtrain = x[fit_idx].astype(np.float32) / 255.
    for j, seed in enumerate(CFG['cnn_seeds']):
        torch.manual_seed(seed)
        tag = 'cnn' if j == 0 else 'aux'
        model = cmod.CNN(len(np.unique(y)))
        if args.frozen:
            model.load_state_dict(torch.load(frozen/f'{name}_{tag}.pt', map_location='cpu'))
        else:
            model, avg = fit(model, xtrain, y[fit_idx], CFG['cifar_epochs'], seed, dev, True, xrest if j == 1 else None)
            if avg is not None:
                saved.update(chen_summary(avg, y[rest]))
            torch.save(model.cpu().state_dict(), out / f'{name}_{tag}.pt')
        saved[tag], saved[tag + '_logits'] = infer(model.to(dev), xrest, dev)
    if args.frozen:
        saved.update(dict(np.load(frozen/f'{name}_chen.npz')))
    np.savez_compressed(out / f'{name}_features.npz', **saved)


if __name__ == '__main__':
    main()
