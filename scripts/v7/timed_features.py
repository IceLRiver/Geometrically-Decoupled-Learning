"""Actual-cadence preprocessing context; model weights stay unchanged."""
from contextlib import contextmanager
import sys
import numpy as np
import benchmark_data
ORIGINAL=benchmark_data.kinematics
@contextmanager
def cadence(dt):
 def timed(X,degree=2):
  f,dc,dl=ORIGINAL(X,degree);f=f.astype(float);f[:,18:21]/=dt;f[:,21:24]/=dt**2;f[:,24:27]/=dt;f[:,27:30]/=dt**2
  return f.astype('float32'),dc.astype(float)/dt,dl.astype(float)/dt
 saved=[]
 for name in ['benchmark_data','stable_inference','train_recent','inference_v7','data_v5','data_v7','neural_predict']:
  module=sys.modules.get(name)
  if module is not None and hasattr(module,'kinematics'):saved.append((module,module.kinematics));module.kinematics=timed
 try:yield
 finally:
  for module,old in saved:module.kinematics=old
