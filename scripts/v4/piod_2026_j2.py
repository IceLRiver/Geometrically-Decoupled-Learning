"""Near-Earth J2 adaptation of Scorsoglio et al., Acta Astronautica 238 (2026).
DOI: 10.1016/j.actaastro.2025.09.002. Not the authors' released software.
The broader X-GEO force model is reduced to central gravity + dominant J2;
unknown spacecraft SRP parameters are not invented.
"""
import numpy as np
from scipy.optimize import least_squares
from physics_baselines import MU,basis,propagate,endpoint_fit
J2=1.08263e-3;RE=6378.137

def acceleration_and_jacobian(p):
 p=np.asarray(p);r2=np.sum(p*p,axis=1);rn=np.sqrt(r2);z=p[:,2];q=z*z/r2;k=1.5*J2*MU*RE**2/rn**5;c=np.array([1.,1.,3.]);F=p*(5*q[:,None]-c);a=-MU*p/rn[:,None]**3+k[:,None]*F;I=np.eye(3)
 central=-MU/rn[:,None,None]**3*(I-3*p[:,:,None]*p[:,None,:]/r2[:,None,None]);dk=-5*k[:,None]*p/r2[:,None];dq=-2*z[:,None]**2*p/r2[:,None]**2;dq[:,2]+=2*z/r2
 jac=central+F[:,:,None]*dk[:,None,:]+k[:,None,None]*(I*(5*q[:,None,None]-c[None,:,None])+5*p[:,:,None]*dq[:,None,:])
 return a,jac

def solve_j2(X,seed=42,verify_jacobian=False):
 X=np.array(X,float,copy=True);X[:,3:]/=np.linalg.norm(X[:,3:],axis=1,keepdims=True);_,init,initial_score=endpoint_fit(X);rng=np.random.default_rng(seed);n=40;w=rng.uniform(-1,1,n);b=rng.uniform(-1,1,n);tc=np.linspace(0,39,50);tm=np.arange(40.);H,D,DD=basis(2*tc/39-1,w,b,2/39);Hm,Dm,_=basis(2*tm/39-1,w,b,2/39)
 _,s,V=np.linalg.svd(np.r_[H,39*D,39**2*DD],full_matrices=False);keep=s>s[0]*1e-12;W=V[keep].T/s[keep];H,D,DD,Hm,Dm=[a@W for a in [H,D,DD,Hm,Dm]];K=H.shape[1];p0,_=propagate(*init,tc);B=np.linalg.lstsq(H,p0,rcond=1e-12)[0];LOS=X[:,3:];obs=np.c_[LOS[:,1]/np.linalg.norm(LOS[:,:2],axis=1),LOS[:,0]/np.linalg.norm(LOS[:,:2],axis=1),LOS[:,2]];I=np.eye(3)
 def fun(beta,jac=False):
  beta=beta.reshape(K,3);p=H@beta;force,G=acceleration_and_jacobian(p);dyn=DD@beta-force;rel=Hm@beta-X[:,:3];rho=np.maximum(np.linalg.norm(rel,axis=1),1.);u=rel/rho[:,None];x,y,z=rel.T;h=np.maximum(np.hypot(x,y),1.);pred=np.c_[y/h,x/h,z/rho]
  if not jac:return np.r_[dyn.ravel(),(pred-obs).ravel()]
  Q=np.zeros((40,3,3));Q[:,0,0]=-x*y/h**3;Q[:,0,1]=x*x/h**3;Q[:,1,0]=y*y/h**3;Q[:,1,1]=-x*y/h**3;Q[:,2,:]=(I[2]-u[:,2,None]*u)/rho[:,None]
  Jd=np.einsum('nk,ij->nikj',DD,I)-np.einsum('nij,nk->nikj',G,H);Jm=np.einsum('nij,nk->nikj',Q,Hm);return np.r_[Jd.reshape(-1,K*3),Jm.reshape(-1,K*3)]
 if verify_jacobian:
  d=rng.normal(size=B.size);d/=np.linalg.norm(d);fd=(fun(B.ravel()+.01*d)-fun(B.ravel()-.01*d))/.02;an=fun(B.ravel(),True)@d;assert np.linalg.norm(fd-an)/np.linalg.norm(an)<1e-3
 fit=least_squares(fun,B.ravel(),jac=lambda b:fun(b,True),method='lm',max_nfev=30,ftol=1e-10,xtol=1e-10,gtol=1e-10);beta=fit.x.reshape(K,3);return np.r_[Hm[-1]@beta,Dm[-1]@beta],{'force_model':'central + J2','J2':J2,'Earth_radius_km':RE,'rank':K,'evaluations':fit.nfev,'loss':float(np.mean(np.abs(fun(fit.x)))),'warm_start_angular_mse':initial_score}
