from pathlib import Path
from collections import defaultdict
import json,csv,numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
 data=json.loads((ROOT/'data/fig-2_neighbour_count.json').read_text(encoding='utf8'))['cells'];bins=defaultdict(lambda:np.zeros(5,dtype=int))
 for r in data:
  if r['flag']==1:bins[min(r['m'],5)]+=np.array([r[k] for k in ('n','local_cont','local_clean','margin_cont','margin_clean')])
 out=ROOT/'outputs';out.mkdir(exist_ok=True)
 with (out/'neighbour_counts.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.writer(f);w.writerow(['corrupted_neighbours','n','local_cont','local_clean','margin_cont','margin_clean'])
  for k in sorted(bins):w.writerow([k,*bins[k]])
 x=np.arange(6);fig,ax=plt.subplots(figsize=(4.5,3));styles=[('local score, corrupted pool','#0072b2','-'),('local score, clean pool','#0072b2','--'),('margin score, corrupted pool','#d55e00','-'),('margin score, clean pool','#d55e00','--')]
 for i,(label,color,ls) in enumerate(styles,1):
  y=[100*bins[k][i]/bins[k][0] for k in x];ax.plot(x,y,marker='o',markersize=3,color=color,ls=ls,label=label)
 ax.set_xticks(x,['0','1','2','3','4','5+']);ax.set_xlabel('corrupted labels among the 10 nearest neighbors');ax.set_ylabel('injected errors caught\nat 10% budget (%)');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.23));fig.tight_layout();fig.savefig(out/'neighbour.pdf')
 assert bins[1][0]==31964 and bins[5][0]==1629
 print('Figure 2: pooled counts and rates reproduced')
if __name__=='__main__':main()
