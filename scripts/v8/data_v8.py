from paths import BASE,R
import json
from data_v5 import load_data,kinematics,observer_frame
def protocol():return json.loads((BASE/'data/benchmark_protocol_v8.json').read_text(encoding='utf-8'))
