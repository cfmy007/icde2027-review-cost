from pathlib import Path
import shutil,subprocess,sys,tempfile

root=Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix='review-replay-') as temp:
    dest=Path(temp)/'artifact'
    shutil.copytree(root,dest)
    for study,subdir in [('text','new_crowd'),('main','historical')]:
        for folder in (dest/'raw_rankings'/study/'generated').iterdir():
            target=dest/'work/decision_closure_20261111/inputs'/subdir/folder.name
            for filename in ('paired.csv','target.npz'):
                shutil.copy2(folder/filename,target/filename)
    subprocess.run([sys.executable,'reproduce.py'],cwd=dest,check=True)
