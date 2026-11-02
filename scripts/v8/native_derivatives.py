import numpy as np
def derivatives(X,B,degree):
 N=X.shape[1];T=float(N-1);tau=(np.arange(N)-T)/T;A=np.stack([tau**j for j in range(degree+1)],1);P=np.linalg.pinv(A);R=np.einsum('nbi,nti->ntb',B,X[:,:,:3]);L=np.einsum('nbi,nti->ntb',B,X[:,:,3:]);dc=np.einsum('i,nij->nj',np.linalg.pinv(np.stack([tau**j for j in range(4)],1))[1],R)/T;dl=np.einsum('i,nij->nj',P[1],L)/T;last=L[:,-1];dl-=np.sum(dl*last,1,keepdims=True)*last;return dc,dl
