"""Cadence-matched alpha-IOD (+OR) reimplementation, TAES 2025.
DOI 10.1109/TAES.2025.3563147. Input sampling is one second, so N_states=40
and BOTH loss components use 40 epochs, consistent with Eq. (14)'s dependence
on measurement cadence. The source example's 0.2-s cadence is not imposed
on these 1-s inputs. Coarse bounds are widened for the benchmark orbit families.
This is model inference code and does not construct a dataset.
"""
import numpy as np
from scipy.optimize import least_squares
from physics_baselines import MU,basis,propagate,lambert

def abc_coarse_cadence(X,rng,population=100,iterations=100):
 """Artificial bee colony endpoint-range search used by alpha-IOD."""
 R=X[:,:3];L=X[:,3:];P=population;bounds=(1.,100000.)
 def initial(n):
  # Log-spaced range proposals avoid starving near-Earth solutions when the
  # physical search interval also contains high-altitude targets.
  a=np.exp(rng.uniform(np.log(bounds[0]),np.log(bounds[1]),n));r=R[0]+a[:,None]*L[0];base=(R[-1]-R[0])/39+a[:,None]*(L[-1]-L[0])/39;den=r@L[0];den=np.where(np.abs(den)>1,den,1);rate=-np.sum(r*base,axis=1)/den
  return np.clip(np.c_[a,a+39*(rate+rng.uniform(-.5,.5,n))],*bounds)
 def fitness(q):
  r=R[0]+q[:,0,None]*L[0];rf=R[-1]+q[:,1,None]*L[-1];v,_=lambert(r,rf)
  pp,_=propagate(r[:,None,:],v[:,None,:],np.arange(1,39)[None,:]);u=pp-R[None,1:39];u/=np.linalg.norm(u,axis=-1,keepdims=True)
  # For unit vectors, 1-u.l = ||u-l||^2/2. The latter avoids cancellation
  # and float32 unit-norm error swamping a short-arc objective near 1e-10.
  J=.5*np.sum((u-L[None,1:39])**2,axis=(1,2));speed=np.linalg.norm(v,axis=1);rn=np.linalg.norm(r,axis=1);rfn=np.linalg.norm(rf,axis=1)
  energy=speed**2/2-MU/rn;semimajor=-MU/(2*np.minimum(energy,-1e-20));ecc=np.linalg.norm(np.cross(v,np.cross(r,v))/MU-r/rn[:,None],axis=1)
  penalty=np.maximum(6378.137+250-rn,0)/1000+np.maximum(rn-55000,0)/1000+np.maximum(6378.137+250-rfn,0)/1000+np.maximum(rfn-55000,0)/1000+np.maximum(energy,0)+np.maximum(ecc-.3,0)+np.maximum(semimajor-60000,0)/10000+np.maximum(6378.137+250-semimajor,0)/10000
  bad=(penalty>0)|(~np.isfinite(J))
  return np.where(bad,1e3+np.nan_to_num(penalty,nan=1e6),np.maximum(J,0))
 q=initial(P);J=fitness(q);trials=np.zeros(P,int);global_index=int(np.argmin(J));global_q=q[global_index].copy();global_J=float(J[global_index])
 for it in range(iterations):
  for phase in range(2):
   probability=1/(1+np.maximum(J,0));probability/=probability.sum()
   parents=np.arange(P) if phase==0 else rng.choice(P,P,p=probability)
   k=(parents+rng.integers(1,P,P))%P;j=rng.integers(0,2,P);cand=q[parents].copy();cand[np.arange(P),j]+=rng.uniform(-1,1,P)*(q[parents,j]-q[k,j]);cand=np.clip(cand,*bounds);jc=fitness(cand)
   for a,parent in enumerate(parents):
    if jc[a]<J[parent]:q[parent]=cand[a];J[parent]=jc[a];trials[parent]=0
    else:trials[parent]+=1
  k=int(np.argmin(J))
  if J[k]<global_J:global_J=float(J[k]);global_q=q[k].copy()
  scout=trials>50
  if scout.any():q[scout]=initial(scout.sum());J[scout]=fitness(q[scout]);trials[scout]=0
  if global_J<1e-10:break
 r=R[0]+global_q[0]*L[0];rf=R[-1]+global_q[1]*L[-1];v,_=lambert(r,rf)
 return (r,v),global_J,it+1

def alpha_pielm(X,init,rng,version='2026',neurons=None,iterations=30,physics_weight=1.,initial_threshold=np.inf,stop_threshold=0.,verify_jacobian=False):
 """Analytic-Jacobian tanh output-layer fit. SVD removes unsupported directions."""
 modern=version in ['2023','2026'];n=neurons or (40 if modern else 101);tc=np.linspace(0,39,50 if modern else 40);tm=np.arange(40,dtype=float)
 if version=='2023':physics_weight*=39**2/6378.137
 w=rng.uniform(-1,1,n) if modern else rng.uniform(0,.01,n);b=rng.uniform(-1,1,n) if modern else rng.uniform(0,.01,n)
 scale=2/39 if modern else 1.;arg=lambda t:2*t/39-1 if modern else t
 H,D,DD=basis(arg(tc),w,b,scale);Hm,Dm,_=basis(arg(tm),w,b,scale)
 # Whiten the joint position/velocity/acceleration basis, preserving its numerical span.
 _,s,V=np.linalg.svd(np.r_[H,39*D,39**2*DD],full_matrices=False);keep=s>s[0]*1e-12
 # alpha-IOD's Moore--Penrose update must retain the Euclidean norm of
 # the original output coefficients. An ORTHOGONAL compression does so;
 # whitening would change that norm and can amplify weak range directions.
 W=V[keep].T/s[keep] if modern else V[keep].T
 H,D,DD,Hm,Dm=[a@W for a in [H,D,DD,Hm,Dm]];K=H.shape[1]
 p0,_=propagate(*init,tc);B=np.linalg.lstsq(H,p0,rcond=1e-12)[0]
 R=X[:,:3];LOS=X[:,3:];eye=np.eye(3)
 if modern:obs=np.c_[LOS[:,1]/np.linalg.norm(LOS[:,:2],axis=1),LOS[:,0]/np.linalg.norm(LOS[:,:2],axis=1),LOS[:,2]]
 else:obs=LOS
 def fun(beta,jac=False):
  beta=beta.reshape(K,3);p=H@beta;rn=np.maximum(np.linalg.norm(p,axis=1),1.);dyn=DD@beta+MU*p/rn[:,None]**3
  rel=Hm@beta-R;rho=np.maximum(np.linalg.norm(rel,axis=1),1.);u=rel/rho[:,None]
  if modern:
   x,y,z=rel.T;h=np.maximum(np.hypot(x,y),1.);pred=np.c_[y/h,x/h,z/rho]
   Q=np.zeros((40,3,3));Q[:,0,0]=-x*y/h**3;Q[:,0,1]=x*x/h**3;Q[:,1,0]=y*y/h**3;Q[:,1,1]=-x*y/h**3;Q[:,2,:]=(eye[2]-u[:,2,None]*u)/rho[:,None]
  else:pred=u;Q=(eye-u[:,:,None]*u[:,None,:])/rho[:,None,None]
  if not jac:return np.r_[(physics_weight*dyn).ravel(),(pred-obs).ravel()]
  G=MU/rn[:,None,None]**3*(eye-3*p[:,:,None]*p[:,None,:]/rn[:,None,None]**2)
  Jd=np.einsum('nk,ij->nikj',DD,eye)+np.einsum('nij,nk->nikj',G,H)
  Jm=np.einsum('nij,nk->nikj',Q,Hm)
  return np.r_[physics_weight*Jd.reshape(-1,K*3),Jm.reshape(-1,K*3)]
 initial_loss=float(np.mean(np.abs(fun(B.ravel()))))
 if verify_jacobian:
  direction=rng.normal(size=B.size);direction/=np.linalg.norm(direction);step=.01;finite=(fun(B.ravel()+step*direction)-fun(B.ravel()-step*direction))/(2*step);analytic=fun(B.ravel(),True)@direction;relative=np.linalg.norm(finite-analytic)/max(np.linalg.norm(analytic),1e-20)
  if relative>1e-3:raise AssertionError(f'Jacobian directional check relative error {relative}')
 if not modern:
  beta=B.ravel();best_beta=beta.copy();best_loss=initial_loss
  for step in range(iterations):
   residual=fun(beta);loss=float(np.mean(np.abs(residual)))
   if loss<best_loss:best_loss=loss;best_beta=beta.copy()
   if loss<=stop_threshold:break
   update=np.linalg.lstsq(fun(beta,True),-residual,rcond=1e-12)[0];beta+=update
  loss=float(np.mean(np.abs(fun(beta))))
  if loss<best_loss:best_beta=beta.copy();best_loss=loss
  beta=best_beta.reshape(K,3);state=np.r_[Hm[-1]@beta,Dm[-1]@beta]
  return state,{'loss':best_loss,'initial_loss':initial_loss,'rank':K,'evaluations':step+1,'success':best_loss<=stop_threshold}
 fit=least_squares(fun,B.ravel(),jac=lambda b:fun(b,True),method='lm',max_nfev=iterations,ftol=1e-10,xtol=1e-10,gtol=1e-10)
 beta=fit.x.reshape(K,3);state=np.r_[Hm[-1]@beta,Dm[-1]@beta];loss=float(np.mean(np.abs(fun(fit.x))))
 return state,{'loss':loss,'initial_loss':initial_loss,'rank':K,'evaluations':fit.nfev,'success':bool(fit.success)}

def alpha_iod_cadence(X,seed=42,runs=100):
 X=np.array(X,dtype=float,copy=True);X[:,3:]/=np.linalg.norm(X[:,3:],axis=1,keepdims=True)
 rng=np.random.default_rng(seed);init,j,it=abc_coarse_cadence(X,rng);states=[];losses=[];attempts=0;fail=0
 coarse,_=propagate(*init,np.arange(40));rel=coarse-X[:,:3];u=rel/np.linalg.norm(rel,axis=1,keepdims=True);coarse_loss=float(np.mean(np.abs(u-X[:,3:]))/2)
 floor=1e-8 if coarse_loss>=1e-4 else (5e-9 if coarse_loss>=1e-7 else 1e-9)
 threshold=5*coarse_loss if coarse_loss>=1e-2 else coarse_loss if coarse_loss>=1e-4 else 2.5e-3
 while len(states)<runs and attempts<runs*20:
  attempts+=1
  # Choose a random basis whose coarse-fit initial loss meets ALFA Table I.
  basis_seed=None;best_initial=np.inf
  for trial in range(100):
   candidate=int(rng.integers(2**31));_,pre=alpha_pielm(X,init,np.random.default_rng(candidate),'2025',iterations=1,stop_threshold=np.inf)
   if pre['initial_loss']<best_initial:best_initial=pre['initial_loss'];basis_seed=candidate
   if best_initial<threshold:break
  s,d=alpha_pielm(X,init,np.random.default_rng(basis_seed),'2025',iterations=30,stop_threshold=floor);loss=d['loss']
  if np.isfinite(s).all() and loss<floor:
   states.append(s);losses.append(loss);floor=max(1e-10,min(floor,loss));fail=0
  else:
   fail+=1
   if fail>=10:floor*=10;fail=0
 if not states:raise RuntimeError('alpha-IOD: no accepted ELM fit')
 states=np.array(states);losses=np.array(losses);best=losses.min();factor=1.5 if best>=1e-8 else 2 if best>=1e-9 else 5 if best>=1e-10 else 50 if best>=1e-11 else 100;mask=losses<=factor*best
 radii=np.linalg.norm(states[mask,:3],axis=1);keep=np.abs(radii-radii.mean())<=max(radii.std(),1e-10);selected=states[mask][keep]
 return selected.mean(0),{'accepted_runs':len(states),'requested_runs':runs,'attempts':attempts,'abc_iterations':it,'abc_fitness':j,'coarse_loss':coarse_loss,'best_loss':best,'retained':len(selected),'outlier_filter':True}
