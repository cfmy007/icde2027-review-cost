"""Prepare official CIFAR inputs or rebuild CASIA caches from supplied GNT zips."""
import argparse
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

HERE=Path(__file__).resolve().parent
CIFAR='https://www.cs.toronto.edu/~kriz'
CIFARN='https://raw.githubusercontent.com/UCSC-REAL/cifar-10-100n/49df7d8a69e355470c77c1c2f2424916325a394b/data'


def download(url, dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():
        partial=dst.with_suffix(dst.suffix+'.part')
        urlretrieve(url,partial)
        partial.replace(dst)


def public(base, dataset):
    for folder, name, clean_dir in [('cifar10','CIFAR-10','cifar-10-batches-py'),
                                    ('cifar100','CIFAR-100','cifar-100-python')]:
        if dataset not in ('all',folder):
            continue
        raw=base/'data/raw'/folder
        if not (raw/clean_dir).exists():
            tar=raw/(name.lower()+'-python.tar.gz')
            download(CIFAR+'/'+tar.name,tar)
            with tarfile.open(tar) as f:
                f.extractall(raw, filter='data')
        pt=name+'_human.pt'
        download(CIFARN+'/'+pt,base/'data/raw'/(folder+'n')/pt)


def casia(base, gnt, rebuild_train):
    from prepare_features import import_script
    old=import_script(base/'scripts/29_curated_bench.py')
    sys.path.insert(0,str(base/'src'))
    from hcc.index import build_cache
    chars={c for pair in old.PAIRS for c in pair}
    dst=base/'data/processed'
    build_cache(gnt/'Gnt1.1Test.zip',dst/'hwdb11_test_32.npz')
    if rebuild_train:
        for i in (1,2):
            build_cache(gnt/f'Gnt1.1TrainPart{i}.zip',dst/f'hwdb11_train{i}_100_32.npz',only_chars=chars)



if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base',type=Path,default=HERE/'base')
    p.add_argument('--cifar',choices=['all','cifar10','cifar100'])
    p.add_argument('--gnt',type=Path)
    p.add_argument('--rebuild-train',action='store_true')
    a=p.parse_args()
    if a.cifar:
        public(a.base,a.cifar)
    if a.gnt:
        casia(a.base,a.gnt,a.rebuild_train)
