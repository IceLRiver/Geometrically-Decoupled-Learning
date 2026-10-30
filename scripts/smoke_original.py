"""Frozen original-model smoke check; not an independent test benchmark."""
from pathlib import Path
import sys,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models import OrbitPhysKAN
state=torch.load(ROOT/'checkpoints/original_orbit_physkan.pth',map_location='cpu',weights_only=True)
keys=['cam_mean','cam_std','los_mean','los_std','rho_mean','rho_std','pos_mean','pos_std','vel_mean','vel_std']
norm={k:state[k] for k in keys};norm['rho_mean']=float(norm['rho_mean']);norm['rho_std']=float(norm['rho_std'])
model=OrbitPhysKAN(norm_params=norm);model.load_state_dict(state);model.eval()
data=np.load(ROOT/'data/legacy_simulation/block_000.npz');stats=np.load(ROOT/'data/legacy_simulation/norm_stats_legacy.npz')
x=(data['X'][:16]-stats['X_mean'])/(stats['X_std']+1e-12)
with torch.no_grad():out=model(torch.tensor(x,dtype=torch.float32))
assert all(torch.isfinite(v).all() for v in out.values())
print('PASS: original checkpoint produces finite range, position and velocity for 16 legacy inputs. This is only a smoke check.')
