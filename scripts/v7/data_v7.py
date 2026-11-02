from pathlib import Path
import sys,os,json
BASE=Path(__file__).resolve().parents[2]
ROOT=Path(os.environ.get('ORBIT_RUN_ROOT',str(BASE/'runs/v7')))
(ROOT/'evidence').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(BASE/'scripts/v5'))
from data_v5 import load_data,kinematics,observer_frame
def protocol():return json.loads((BASE/'data/benchmark_protocol_v7.json').read_text(encoding='utf-8'))
freeze_protocol=protocol
