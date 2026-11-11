from pathlib import Path
from collections import defaultdict
import csv,numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
ORDER=('NC-CNN','NC-PCA','NC-RP','NC-Aux','AQuA-ID-CNN','AQuA-ID-Aux','AQuA-CM-CNN','AQuA-CM-Aux','Chen-ID-mask','Chen-ID-hard','Uniform')
LABEL=('NC-CNN','NC-PCA','NC-RP','NC-Aux','ID-CNN','ID-Aux','CM-CNN','CM-Aux','Chen mask','Chen hard','Uniform')

def main():
 grouped=defaultdict(list)
 with (ROOT/'data/fig-3_c100_regret.csv').open(encoding='utf8',newline='') as f:
  for r in csv.DictReader(f):
   if abs(float(r['rate'])-.1)<1e-9 and abs(float(r['budget'])-.1)<1e-9:grouped[r['source'],r['target']].append(float(r['regret_pp'])*100)
 matrix=np.array([[np.mean(grouped[a,b]) for b in ORDER] for a in ORDER]);out=ROOT/'outputs';out.mkdir(exist_ok=True)
 with (out/'regret_matrix.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['source',*ORDER])
  for name,row in zip(ORDER,matrix):w.writerow([name,*row])
 fig,ax=plt.subplots(figsize=(4.6,4.5));im=ax.imshow(matrix,cmap='Blues');ax.set_xticks(np.arange(11),LABEL,rotation=90,fontsize=7);ax.set_yticks(np.arange(11),LABEL,fontsize=7)
 for i in range(11):
  for j in range(11):
   v=matrix[i,j];ax.text(j,i,str(round(v)) if v>0 else '.',ha='center',va='center',fontsize=5.5,color='white' if v>matrix.max()*.5 else 'black')
 ax.set_xlabel('Target injection protocol');ax.set_ylabel('Selection protocol');fig.colorbar(im,ax=ax,label='Missed fixes per 10,000');fig.tight_layout();fig.savefig(out/'regret_matrix.pdf')
 assert abs(matrix[0,1]-293.088)<.01
 print('Figure 3: CIFAR-100 target regret matrix reproduced')
if __name__=='__main__':main()
