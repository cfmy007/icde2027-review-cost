from pathlib import Path
import csv,json,subprocess,sys
ROOT=Path(__file__).resolve().parent

def command(script,*args):
 p=ROOT/script
 result=subprocess.run([sys.executable,str(p),*args],cwd=p.parent,text=True,capture_output=True,encoding='utf8')
 if result.returncode:
  print(result.stdout);print(result.stderr,file=sys.stderr);raise SystemExit(result.returncode)
 print('PASS',script,flush=True)

def read(paths):
 out={}
 for path in paths:
  with (ROOT/path).open(encoding='utf8',newline='') as f:
   for r in csv.DictReader(f):out[r['dataset']]=r
 return out

def check_outputs():
 new=read(['work/reference_pairing/sequential_queues/equal_risk_comparison/summary.csv','work/reference_pairing/sequential_queues/equal_risk_extra/summary.csv'])
 old=read(['data/table-10_equal_risk_summary.csv','data/table-10_equal_risk_extra.csv'])
 assert len(new)==len(old)==33
 for name in new:
  for field in ('default','interleave','fixed','feedback'):
   assert abs(float(new[name][field])-float(old[name][field]))<1e-8,(name,field)
 print('PASS 33 review-policy summaries match the supplied measurements')
 for dataset in ('c100','wiki'):
  for learner in ('linear','mlp'):
   a=json.loads((ROOT/'data'/f'table-8_{dataset}_{learner}.json').read_text(encoding='utf8'))
   b=json.loads((ROOT/'downstream'/f'equal_cost_{dataset}_{learner}_summary.json').read_text(encoding='utf8'))
   assert a['policy']==b['policy'] and a['contrasts']==b['contrasts'],(dataset,learner)
 print('PASS four downstream policy tables and paired bootstrap results match')
 for folder,filename in (('review_bridge','human_bridge_cnn_summary.json'),('review_bridge_linear','human_bridge_linear_summary.json')):
  a=json.loads((ROOT/'data'/filename).read_text(encoding='utf8'))
  b=json.loads((ROOT/'downstream/human_bridge'/folder/'summary.json').read_text(encoding='utf8'))
  assert a==b,folder
 print('PASS both CIFAR-10 human-review bootstrap summaries match')

if __name__=='__main__':
 command('check_saved_tables.py')
 command('replay_primary.py')
 command('replay_source_selection.py')
 command('replay_crossed_tables.py')
 command('work/reference_pairing/sequential_queues/equal_risk_comparison/run.py')
 command('work/reference_pairing/sequential_queues/equal_risk_extra/run.py')
 command('replay_cap_sweep.py')
 command('replay_neighbours.py')
 command('replay_regret_figure.py')
 command('work/reference_pairing/sequential_queues/paired_displacement/run.py')
 command('work/reference_pairing/sequential_queues/paired_diagnosis/run.py')
 command('work/reference_pairing/sequential_queues/paired_bias_correction/run.py')
 command('downstream/verify_equal_cost_selection.py')
 command('downstream/verify_equal_cost_class_match.py')
 command('downstream/replay_downstream_points.py')
 command('downstream/replay_text_curves.py')
 command('downstream/human_bridge/review_bridge_summary.py','--out',str(ROOT/'downstream/human_bridge/review_bridge'))
 command('downstream/human_bridge/review_bridge_linear_summary.py','--out',str(ROOT/'downstream/human_bridge/review_bridge_linear'))
 for dataset in ('c100','wiki'):
  for learner in ('linear','mlp'):
   command('downstream/summarize_equal_cost_selection.py',dataset,learner,'--draws','2000')
 check_outputs()
 print('Replay complete. See README.md for inputs and scope.')
