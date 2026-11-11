from pathlib import Path
import subprocess,sys
root=Path(__file__).resolve().parent
names=[p.name.removesuffix('.jsonl.gz') for p in sorted((root/'source').glob('*.jsonl.gz'))]
for script in ('prepare.py','generate.py','compare.py'):
    subprocess.run([sys.executable,str(root/script),*(names if script!='compare.py' else [])],check=True)
