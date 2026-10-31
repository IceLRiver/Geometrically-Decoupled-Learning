"""Executed orbit solvers; input positions km, time s, output terminal km/km s-1.

Research reimplementations, not the authors' software. No target state is accepted.
ORBDTOOLS 0.2.1 supplies the classical polynomial solvers (MIT license).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from scipy.optimize import least_squares
MU=398600.4418

def cs(z):
 z=np.asarray(z);small=np.abs(z)<1e-5;p=np.sqrt(np.maximum(z,1e-30));n=np.sqrt(np.maximum(-z,1e-30))
 with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
  C=np.where(z>0,(1-np.cos(p))/p**2,(np.cosh(n)-1)/n**2)
  S=np.where(z>0,(p-np.sin(p))/p**3,(np.sinh(n)-n)/n**3)
 return np.where(small,.5-z/24+z*z/720,C),np.where(small,1/6-z/120+z*z/5040,S)

def propagate(r,v,t):
 r=np.asarray(r);v=np.asarray(v);t=np.asarray(t);r0=np.linalg.norm(r,axis=-1);rv=np.sum(r*v,axis=-1);alpha=2/r0-np.sum(v*v,axis=-1)/MU;chi=np.sqrt(MU)*t/r0
 for _ in range(18):
  z=alpha*chi**2;C,S=cs(z);F=rv/np.sqrt(MU)*chi**2*C+(1-alpha*r0)*chi**3*S+r0*chi-np.sqrt(MU)*t
  d=rv/np.sqrt(MU)*chi*(1-z*S)+(1-alpha*r0)*chi**2*C+r0;step=F/d;chi-=np.clip(step,-1000,1000)
 C,S=cs(alpha*chi**2);f=1-chi**2*C/r0;g=t-chi**3*S/np.sqrt(MU);p=f[...,None]*r+g[...,None]*v;rn=np.linalg.norm(p,axis=-1)
 fd=np.sqrt(MU)/(rn*r0)*(alpha*chi**3*S-chi);gd=1-chi**2*C/rn
 return p,fd[...,None]*r+gd[...,None]*v

def lambert(r1,r2,dt=39.):
 a=np.linalg.norm(r1,axis=-1);b=np.linalg.norm(r2,axis=-1);A=np.sqrt(np.maximum(a*b+np.sum(r1*r2,axis=-1),1e-20));lo=np.full_like(a,-4*np.pi**2);hi=np.full_like(a,4*np.pi**2-1e-6)
 for _ in range(58):
  z=(lo+hi)/2;C,S=cs(z);y=a+b+A*(z*S-1)/np.sqrt(C);time=((np.maximum(y,0)/C)**1.5*S+A*np.sqrt(np.maximum(y,0)))/np.sqrt(MU)
  low=(y<0)|(time<dt);lo=np.where(low,z,lo);hi=np.where(low,hi,z)
 f=1-y/a;g=A*np.sqrt(np.maximum(y,1e-30)/MU);gd=1-y/b
 return (r2-f[...,None]*r1)/g[...,None],(gd[...,None]*r2-r1)/g[...,None]

def angular_score(p,X):
 u=p-X[:,:3];u/=np.linalg.norm(u,axis=-1,keepdims=True);return np.mean(np.sum((u-X[:,3:])**2,axis=-1),axis=-1)

def endpoint_fit(X,starts=(500,2000,8000,25000,45000)):
 """All-observation two-endpoint Lambert least squares; common warm start only."""
 X=np.asarray(X,float);best=None
 def res(q):
  r0=X[0,:3]+q[0]*X[0,3:];rf=X[-1,:3]+q[1]*X[-1,3:];v,_=lambert(r0,rf)
  p,_=propagate(r0,v,np.arange(40));u=p-X[:,:3];u/=np.linalg.norm(u,axis=1,keepdims=True);return (u-X[:,3:]).ravel()
 for rho in starts:
  fit=least_squares(res,[rho,rho],bounds=([1,1],[100000,100000]),x_scale='jac',diff_step=1e-4,max_nfev=80,ftol=1e-11,xtol=1e-10,gtol=1e-13)
  score=np.mean(res(fit.x)**2)
  if best is None or score<best[0]:best=(score,fit.x)
 q=best[1];r0=X[0,:3]+q[0]*X[0,3:];rf=X[-1,:3]+q[1]*X[-1,3:];v0,vf=lambert(r0,rf)
 return np.r_[rf,vf],(r0,v0),best[0]

def classical(X,method):
 if method=='Gooding':
  # Two endpoint ranges are solved against TWO transverse middle-LOS
  # components. A scalar 1-cos(theta) residual has only one equation for
  # these two unknowns and is unsuitable for this root problem.
  L=X[19,3:];axis=np.eye(3)[np.argmin(np.abs(L))];e1=np.cross(L,axis);e1/=np.linalg.norm(e1);plane=np.stack([e1,np.cross(L,e1)])
  def residual(q):
   r0=X[0,:3]+q[0]*X[0,3:];rf=X[-1,:3]+q[1]*X[-1,3:];v,_=lambert(r0,rf);p,_=propagate(r0,v,19.);rel=p-X[19,:3];return plane@(rel/np.linalg.norm(rel))
  candidates=[]
  for start in [500.,2000.,8000.,25000.,45000.]:
   fit=least_squares(residual,[start,start],bounds=([1,1],[100000,100000]),x_scale='jac',diff_step=1e-4,max_nfev=80,gtol=1e-13,ftol=1e-11,xtol=1e-11)
   r0=X[0,:3]+fit.x[0]*X[0,3:];rf=X[-1,:3]+fit.x[1]*X[-1,3:];v,vf=lambert(r0,rf);p,_=propagate(r0,v,np.arange(40))
   if np.isfinite(p).all():candidates.append((angular_score(p,X),np.r_[rf,vf]))
  if not candidates:raise RuntimeError('No finite three-LOS Lambert solution')
  return min(candidates,key=lambda z:z[0])[1],{'candidate_count':len(candidates),'implementation':'Gooding-type two-range/three-LOS Lambert root, two tangent residuals; numerical least-squares update'}
 import sys,types,importlib.util
 from pathlib import Path
 if 'orbdtools' not in sys.modules:
  spec=importlib.util.find_spec('orbdtools');pkg=types.ModuleType('orbdtools');pkg.__path__=[str(Path(spec.origin).parent)];sys.modules['orbdtools']=pkg
 from orbdtools.iod.angular.gauss import gauss_estimate
 from orbdtools.iod.angular.laplace import laplace_estimate
 from orbdtools.iod.angular.gooding import gooding_estimate
 from orbdtools.transform.kep_rv_trans import coe2rv
 DU=6378.137;TU=np.sqrt(DU**3/MU);ids=[0,19,39];R=X[ids,:3]/DU;L=X[ids,3:];args=(1.,39/TU,(19/TU,20/TU),L,R)
 if method=='Gauss':eles=gauss_estimate(*args)
 elif method=='Laplace':eles=laplace_estimate(*args)
 else:eles=[gooding_estimate(*args,method='izzo')]
 candidates=[]
 for ele in eles:
  try:
   rv=coe2rv(ele,1.,True);r=rv[:3]*DU;v=rv[3:]*DU/TU;p,_=propagate(r,v,np.arange(40)-19);rf,vf=propagate(r,v,20)
   if np.isfinite(p).all():candidates.append((angular_score(p,X),np.r_[rf,vf]))
  except (ValueError,FloatingPointError):pass
 if not candidates:raise RuntimeError('No finite classical solution')
 return min(candidates,key=lambda z:z[0])[1],{'candidate_count':len(candidates)}

def abc_coarse(X,rng,population=100,iterations=100):
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
  penalty=np.maximum(6378.137+250-rn,0)/1000+np.maximum(rn-55000,0)/1000+np.maximum(6378.137+250-rfn,0)/1000+np.maximum(rfn-55000,0)/1000+np.maximum(energy,0)+np.maximum(ecc-.3,0)+np.maximum(semimajor-60000,0)/10000
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

def basis(t,w,b,scale):
 H=np.tanh(t[:,None]*w+b);D=(1-H**2)*w*scale;DD=-2*H*(1-H**2)*w*w*scale**2;return H,D,DD

def pielm(X,init,rng,version='2026',neurons=None,iterations=30,physics_weight=1.,initial_threshold=np.inf,stop_threshold=0.,verify_jacobian=False):
 """Analytic-Jacobian tanh output-layer fit. SVD removes unsupported directions."""
 modern=version in ['2023','2026'];n=neurons or (40 if modern else 101);tc=np.linspace(0,39,50 if modern else 196);tm=np.arange(40,dtype=float)
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

def alpha_iod(X,seed=42,runs=100):
 X=np.array(X,dtype=float,copy=True);X[:,3:]/=np.linalg.norm(X[:,3:],axis=1,keepdims=True)
 rng=np.random.default_rng(seed);init,j,it=abc_coarse(X,rng);states=[];losses=[];attempts=0;fail=0
 coarse,_=propagate(*init,np.arange(40));rel=coarse-X[:,:3];u=rel/np.linalg.norm(rel,axis=1,keepdims=True);coarse_loss=float(np.mean(np.abs(u-X[:,3:]))/2)
 floor=1e-8 if coarse_loss>=1e-4 else (5e-9 if coarse_loss>=1e-7 else 1e-9)
 threshold=5*coarse_loss if coarse_loss>=1e-2 else coarse_loss if coarse_loss>=1e-4 else 2.5e-3
 while len(states)<runs and attempts<runs*20:
  attempts+=1
  # Choose a random basis whose coarse-fit initial loss meets ALFA Table I.
  basis_seed=None;best_initial=np.inf
  for trial in range(100):
   candidate=int(rng.integers(2**31));_,pre=pielm(X,init,np.random.default_rng(candidate),'2025',iterations=1,stop_threshold=np.inf)
   if pre['initial_loss']<best_initial:best_initial=pre['initial_loss'];basis_seed=candidate
   if best_initial<threshold:break
  s,d=pielm(X,init,np.random.default_rng(basis_seed),'2025',iterations=30,stop_threshold=floor);loss=d['loss']
  if np.isfinite(s).all() and loss<floor:
   states.append(s);losses.append(loss);floor=max(1e-10,min(floor,loss));fail=0
  else:
   fail+=1
   if fail>=10:floor*=10;fail=0
 if not states:raise RuntimeError('alpha-IOD: no accepted ELM fit')
 states=np.array(states);losses=np.array(losses);best=losses.min();factor=1.5 if best>=1e-8 else 2 if best>=1e-9 else 5 if best>=1e-10 else 50 if best>=1e-11 else 100;mask=losses<=factor*best
 radii=np.linalg.norm(states[mask,:3],axis=1);keep=np.abs(radii-radii.mean())<=max(radii.std(),1e-10);selected=states[mask][keep]
 return selected.mean(0),{'accepted_runs':len(states),'requested_runs':runs,'attempts':attempts,'abc_iterations':it,'abc_fitness':j,'coarse_loss':coarse_loss,'best_loss':best,'retained':len(selected),'outlier_filter':True}

def solve(X,method,seed=42):
 X=np.array(X,dtype=float,copy=True);X[:,3:]/=np.linalg.norm(X[:,3:],axis=1,keepdims=True)
 if method in ['Gauss','Laplace','Gooding']:return classical(X,method)
 if method=='alpha-IOD-2025':return alpha_iod(X,seed)
 state,init,score=endpoint_fit(X)
 if method=='Lambert-LS':return state,{'angular_mse':score}
 if method in ['PIOD-2023','PIOD-2026']:
  s,d=pielm(X,init,np.random.default_rng(seed),method[-4:]);d['warm_start']='all-angle Lambert LS';return s,d
 raise ValueError(method)
