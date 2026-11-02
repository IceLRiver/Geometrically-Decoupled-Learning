"""Numerically guarded, observation-only PIOD-family task reimplementations.

No reference state is accepted. The coarse/admissibility protocol and optimization
choices must be reported; these are not original-author software or claimed exact
reproductions. Coordinates km, native epochs s, terminal state km and km/s.
"""
from pathlib import Path
import sys,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
from scipy.optimize import least_squares
from paths import BASE
from physics_baselines import propagate,lambert,basis,MU
from piod_2026_j2 import acceleration_and_jacobian
RE=6378.137

def geometry(X):
 X=np.array(X,float,copy=True);X[:,3:]/=np.linalg.norm(X[:,3:],axis=1,keepdims=True);return X

def feasible(p,v,R,rho_max=10000.,rho_reference=None,trust_radius=None,velocity_reference=None,velocity_trust=None):
 r=np.linalg.norm(p,axis=-1);rho=np.linalg.norm(p-R,axis=-1);energy=np.sum(v*v,axis=-1)/2-MU/r
 trust_ok=True if rho_reference is None or trust_radius is None else np.all(np.abs(rho-rho_reference)<=trust_radius)
 velocity_ok=True if velocity_reference is None or velocity_trust is None else np.all(np.linalg.norm(v-velocity_reference,axis=-1)<=velocity_trust)
 return bool(np.isfinite(p).all() and np.isfinite(v).all() and np.all(r>RE+200) and np.all(r<65000) and np.all(rho>=.1) and np.all(rho<=rho_max) and np.all(energy<0) and trust_ok and velocity_ok)

def coarse(X,prior=None,rho_max=10000.):
 """Multistart two-range Lambert initialization with an explicit survey range."""
 N=len(X);T=float(N-1);R=X[:,:3];L=X[:,3:];starts=[150.,500.,1500.,3500.,6500.,9500.]
 if prior:
  starts+=prior.get('range_initializers',[])
  for radius in prior.get('radius_initializers',[]):
   dot=R[0]@L[0];disc=dot*dot+radius**2-R[0]@R[0]
   if disc>=0:
    for q in [-dot-np.sqrt(disc),-dot+np.sqrt(disc)]:
     if 1<q<rho_max:starts.append(float(q))
 def trajectory(q):
  r=R[0]+q[0]*L[0];rf=R[-1]+q[1]*L[-1];v,_=lambert(r,rf,T);return propagate(r,v,np.arange(N)),(r,v)
 def res(q):
  (p,_),_=trajectory(q);rel=p-R;return (rel/np.linalg.norm(rel,axis=1,keepdims=True)-L).ravel()
 candidates=[]
 for start in starts:
  q=np.clip([start,start],1.01,rho_max-.01)
  fit=least_squares(res,q,bounds=([1,1],[rho_max,rho_max]),x_scale='jac',diff_step=1e-4,max_nfev=100,ftol=1e-11,xtol=1e-11,gtol=1e-13)
  (p,v),init=trajectory(fit.x)
  if feasible(p,v,R,rho_max):candidates.append((np.mean(res(fit.x)**2),init,p,v))
 if not candidates:raise RuntimeError('No admissible Lambert initializer')
 q=min(candidates,key=lambda x:x[0]);return q[1],dict(angular_mse=float(q[0]),admissible_initializers=len(candidates))

def problem(X,init,rng,version,rank_tol=1e-10,physics_weight=1.,j2=False):
 N=len(X);T=float(N-1);modern=version!='2025';nodes=40 if modern else 101;tc=np.linspace(0,T,50 if modern else N);tm=np.arange(N,dtype=float);w=rng.uniform(-1,1,nodes) if modern else rng.uniform(0,.01,nodes);b=rng.uniform(-1,1,nodes) if modern else rng.uniform(0,.01,nodes);scale=2/T if modern else 1.;arg=lambda x:2*x/T-1 if modern else x
 H,D,DD=basis(arg(tc),w,b,scale);Hm,Dm,DDm=basis(arg(tm),w,b,scale)
 _,s,V=np.linalg.svd(np.r_[H,T*D,T*T*DD],full_matrices=False);keep=s>s[0]*rank_tol;W=V[keep].T/s[keep] if modern else V[keep].T;H,D,DD,Hm,Dm,DDm=[x@W for x in [H,D,DD,Hm,Dm,DDm]];K=H.shape[1];p0,_=propagate(*init,tc);B=np.linalg.lstsq(H,p0/RE,rcond=rank_tol)[0];LOS=X[:,3:];R=X[:,:3];I=np.eye(3);wd=physics_weight*(T*T/RE if version=='2023' else 1.)
 obs=np.c_[LOS[:,1]/np.linalg.norm(LOS[:,:2],axis=1),LOS[:,0]/np.linalg.norm(LOS[:,:2],axis=1),LOS[:,2]] if modern else LOS
 def fun(beta,jac=False):
  beta=beta.reshape(K,3);p=RE*H@beta;rel=RE*Hm@beta-R;rho=np.linalg.norm(rel,axis=1);u=rel/np.maximum(rho[:,None],1e-12)
  if j2:force,G=acceleration_and_jacobian(p)
  else:
   rn=np.maximum(np.linalg.norm(p,axis=1),1.);force=-MU*p/rn[:,None]**3;G=-MU/rn[:,None,None]**3*(I-3*p[:,:,None]*p[:,None,:]/rn[:,None,None]**2)
  dyn=RE*DD@beta-force
  if modern:
   x,y,z=rel.T;h=np.maximum(np.hypot(x,y),1e-12);pred=np.c_[y/h,x/h,z/np.maximum(rho,1e-12)];Q=np.zeros((N,3,3));Q[:,0,0]=-x*y/h**3;Q[:,0,1]=x*x/h**3;Q[:,1,0]=y*y/h**3;Q[:,1,1]=-x*y/h**3;Q[:,2,:]=(I[2]-u[:,2,None]*u)/np.maximum(rho[:,None],1e-12)
  else:pred=u;Q=(I-u[:,:,None]*u[:,None,:])/np.maximum(rho[:,None,None],1e-12)
  if not jac:return np.r_[(wd*dyn).ravel(),(pred-obs).ravel()]
  Jd=RE*(np.einsum('nk,ij->nikj',DD,I)-np.einsum('nij,nk->nikj',G,H));Jm=RE*np.einsum('nij,nk->nikj',Q,Hm);return np.r_[wd*Jd.reshape(-1,K*3),Jm.reshape(-1,K*3)]
 def state(beta):
  beta=beta.reshape(K,3);return RE*Hm@beta,RE*Dm@beta
 return B.ravel(),fun,state,K

def fit(X,init,rng,version,rank_tol=1e-10,damping=.01,iterations=100,physics_weight=1.,stop=0.,j2=False,rho_max=10000.,verify=False,trust_radius=None,velocity_reference=None,velocity_trust=None):
 beta,fun,state,K=problem(X,init,rng,version,rank_tol,physics_weight,j2);res=fun(beta);loss=np.mean(np.abs(res));initial_loss=float(loss);best=beta.copy();bestloss=loss;lam=damping;accepted=rejected=0
 if verify:
  d=rng.normal(size=len(beta));d/=np.linalg.norm(d);h=1e-6;fd=(fun(beta+h*d)-fun(beta-h*d))/(2*h);an=fun(beta,True)@d;err=np.linalg.norm(fd-an)/max(np.linalg.norm(an),1e-15);assert err<1e-3,err
 refp,_=propagate(*init,np.arange(len(X)));rho_reference=np.linalg.norm(refp-X[:,:3],axis=1)
 for it in range(iterations):
  if loss<=stop:break
  J=fun(beta,True);U,s,V=np.linalg.svd(J,full_matrices=False);floor=rank_tol*s[0];step=-(V.T*(s/(s*s+lam*s[0]*s[0])))@(U.T@res);trial=beta+step;p,v=state(trial)
  trialres=fun(trial);valid=feasible(p,v,X[:,:3],rho_max,rho_reference,trust_radius,velocity_reference,velocity_trust)
  if valid and np.isfinite(trialres).all() and trialres@trialres<res@res:
   beta=trial;res=trialres;loss=float(np.mean(np.abs(res)));accepted+=1;lam=max(lam/3,1e-14)
   if loss<bestloss:bestloss=loss;best=beta.copy()
  else:rejected+=1;lam=min(lam*10,1e12)
  if np.linalg.norm(step)<1e-12*(1+np.linalg.norm(beta)) or lam>=1e12:break
 p,v=state(best);valid=feasible(p,v,X[:,:3],rho_max,rho_reference,trust_radius,velocity_reference,velocity_trust)
 return np.r_[p[-1],v[-1]],dict(loss=float(bestloss),rank=K,iterations=it+1,accepted_steps=accepted,rejected_steps=rejected,admissible=valid,initial_loss=initial_loss)

def abc(X,rng,rho_max=10000.,seed_state=None,trust_radius=500.):
 N=len(X);T=N-1.;R=X[:,:3];L=X[:,3:];P=100
 if seed_state is not None:
  p0,_=propagate(seed_state[:3],seed_state[3:],-T);center=np.array([np.linalg.norm(p0-R[0]),np.linalg.norm(seed_state[:3]-R[-1])]);lo=np.maximum(1.,center-trust_radius);hi=np.minimum(rho_max,center+trust_radius)
 else:lo=np.ones(2);hi=np.full(2,rho_max)
 def initial(n):
  q=rng.uniform(lo,hi,(n,2))
  if seed_state is None:
   a=np.exp(rng.uniform(np.log(lo[0]),np.log(hi[0]),n));r=R[0]+a[:,None]*L[0];base=(R[-1]-R[0])/T+a[:,None]*(L[-1]-L[0])/T;den=r@L[0];den=np.where(np.abs(den)>1,den,1);rate=-np.sum(r*base,1)/den;q=np.c_[a,a+T*(rate+rng.uniform(-.5,.5,n))]
  return np.clip(q,lo,hi)
 def fitness(q):
  r=R[0]+q[:,0,None]*L[0];rf=R[-1]+q[:,1,None]*L[-1];v,_=lambert(r,rf,T);p,_=propagate(r[:,None],v[:,None],np.arange(1,N-1)[None,:]);rel=p-R[None,1:-1];u=rel/np.linalg.norm(rel,axis=2,keepdims=True);J=.5*np.sum((u-L[None,1:-1])**2,axis=(1,2));rn=np.linalg.norm(r,axis=1);e=np.linalg.norm(np.cross(v,np.cross(r,v))/MU-r/rn[:,None],axis=1);energy=(v*v).sum(1)/2-MU/rn;bad=(rn<RE+200)|(rn>65000)|(energy>=0)|(e>.3)|~np.isfinite(J);return np.where(bad,1e3,np.maximum(J,0))
 q=initial(P)
 if seed_state is not None:q[0]=np.clip(center,lo,hi)
 J=fitness(q);trials=np.zeros(P,int);gb=q[J.argmin()].copy();gj=float(J.min())
 for it in range(100):
  for phase in range(2):
   prob=1/(1+J);prob/=prob.sum();parents=np.arange(P) if phase==0 else rng.choice(P,P,p=prob);k=(parents+rng.integers(1,P,P))%P;j=rng.integers(0,2,P);cand=q[parents].copy();cand[np.arange(P),j]+=rng.uniform(-1,1,P)*(q[parents,j]-q[k,j]);cand=np.clip(cand,lo,hi);jc=fitness(cand)
   for a,parent in enumerate(parents):
    if jc[a]<J[parent]:q[parent]=cand[a];J[parent]=jc[a];trials[parent]=0
    else:trials[parent]+=1
  k=J.argmin()
  if J[k]<gj:gj=float(J[k]);gb=q[k].copy()
  scout=trials>50
  if scout.any():q[scout]=initial(int(scout.sum()));J[scout]=fitness(q[scout]);trials[scout]=0
  if gj<1e-10:break
 r=R[0]+gb[0]*L[0];rf=R[-1]+gb[1]*L[-1];v,_=lambert(r,rf,T);return (r,v),dict(abc_iterations=it+1,abc_fitness=gj,range_bounds=[lo.tolist(),hi.tolist()],seeded=seed_state is not None)

def solve(X,method,seed=42,prior=None,rank_tol=1e-10,damping=.01,physics_weight=1.,runs=100,iterations=100,j2=False,rho_max=10000.,verify=False,init_state=None,trust_radius=None,velocity_trust_km_s=None):
 X=geometry(X);rng=np.random.default_rng(seed)
 if method=='alpha-IOD-2025':init,coarsedetail=abc(X,rng,rho_max,init_state,trust_radius or 500.)
 elif init_state is not None:
  state=np.asarray(init_state,float).copy();rho=np.clip((state[:3]-X[-1,:3])@X[-1,3:],1.,rho_max);state[:3]=X[-1,:3]+rho*X[-1,3:];p0,v0=propagate(state[:3],state[3:],-(len(X)-1));init=(p0,v0);coarsedetail=dict(initialization='frozen training-only RBFNN terminal state projected onto measured LOS')
 else:init,coarsedetail=coarse(X,prior,rho_max)
 velocity_reference=None
 if init_state is not None and velocity_trust_km_s is not None:
  _,velocity_reference=propagate(np.asarray(init_state[:3]),np.asarray(init_state[3:]),np.arange(len(X))-(len(X)-1))
  if method=='alpha-IOD-2025':init=(init[0],velocity_reference[0])
 coarsep,coarsev=propagate(*init,np.arange(len(X)));coarsestate=np.r_[coarsep[-1],coarsev[-1]]
 if method!='alpha-IOD-2025':
  s,d=fit(X,init,rng,method[-4:],rank_tol,damping,iterations,physics_weight,j2=j2,rho_max=rho_max,verify=verify,trust_radius=trust_radius,velocity_reference=velocity_reference,velocity_trust=velocity_trust_km_s)
  if not d['admissible']:s=coarsestate;d['fallback']='admissible Lambert initializer'
  return s,dict(**d,coarse=coarsedetail,rank_tol=rank_tol,initial_damping=damping,physics_weight=physics_weight,j2=j2,range_trust_radius_km=trust_radius,velocity_trust_km_s=velocity_trust_km_s)
 los=coarsep-X[:,:3];los/=np.linalg.norm(los,axis=1,keepdims=True);lmh=float(np.mean(np.abs(los-X[:,3:]))/2);floor=1e-8 if lmh>=1e-4 else 5e-9 if lmh>=1e-7 else 1e-9;threshold=5*lmh if lmh>=1e-2 else lmh if lmh>=1e-4 else 2.5e-3;states=[];losses=[];attempts=fail=0;selected=[]
 while len(states)<runs and attempts<20*runs:
  attempts+=1;best_initial=np.inf;chosen=None
  for trial in range(100):
   key=int(rng.integers(2**31));b,fun,_,_=problem(X,init,np.random.default_rng(key),'2025',rank_tol,physics_weight);l=float(np.mean(np.abs(fun(b))))
   if l<best_initial:chosen=key;best_initial=l
   if best_initial<threshold:break
  s,d=fit(X,init,np.random.default_rng(chosen),'2025',rank_tol,damping,30,physics_weight,stop=floor,rho_max=rho_max,trust_radius=trust_radius,velocity_reference=velocity_reference,velocity_trust=velocity_trust_km_s)
  if d['admissible'] and d['loss']<=floor:states.append(s);losses.append(d['loss']);floor=max(1e-10,min(floor,d['loss']));fail=0
  else:
   fail+=1
   if fail>=10:floor*=10;fail=0
 if not states:return coarsestate,dict(fallback='admissible Lambert initializer',attempts=attempts,accepted_runs=0,coarse=coarsedetail)
 states=np.asarray(states);losses=np.asarray(losses);best=losses.min();factor=1.5 if best>=1e-8 else 2 if best>=1e-9 else 5 if best>=1e-10 else 50 if best>=1e-11 else 100;candidate=states[losses<=factor*best];radii=np.linalg.norm(candidate[:,:3],axis=1);keep=np.abs(radii-radii.mean())<=max(radii.std(),1e-10);selected=candidate[keep];return selected.mean(0),dict(accepted_runs=len(states),attempts=attempts,retained=len(selected),best_loss=float(best),coarse=coarsedetail,stabilization='damped SVD updates and observation-domain admissibility',rank_tol=rank_tol)
