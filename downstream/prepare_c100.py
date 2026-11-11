from pathlib import Path
import argparse,pickle,shutil,numpy as np
ROOT=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('cifar100_dir',type=Path);a=p.parse_args()
 with (a.cifar100_dir/'train').open('rb') as f:train=pickle.load(f,encoding='bytes')
 with np.load(ROOT/'c100_split.npz') as z:fit=z['fit'];rest=z['rest']
 images=np.asarray(train[b'data'],dtype=np.uint8).reshape(-1,3,32,32);labels=np.asarray(train[b'fine_labels'],dtype=np.int64)
 np.savez_compressed(ROOT/'c100_pixels.npz',images=images,y=labels,fit=fit,rest=rest)
 (ROOT/'official').mkdir(exist_ok=True);shutil.copy2(a.cifar100_dir/'test',ROOT/'official/test')
 print('Prepared CIFAR-100 pixels and frozen train split')
if __name__=='__main__':main()
