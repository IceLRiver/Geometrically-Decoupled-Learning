from pathlib import Path
import sys,os
BASE=Path(__file__).resolve().parents[2]
R=Path(os.environ.get('ORBIT_RUN_ROOT',str(BASE/'runs/v8')))
R.mkdir(parents=True,exist_ok=True)
B=BASE
for version in ['v4','v5','v7']:sys.path.insert(0,str(BASE/'scripts'/version))
