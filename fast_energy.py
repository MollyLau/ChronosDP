import numpy as np
from numba import njit
@njit(cache=True)
def integral(x,m,y,P,q):
 j=np.searchsorted(x,q,side='right')-1
 if j<0:j=0
 if j>=len(x):j=len(x)-1
 u=q-x[j]
 return P[j]+y[j]*y[j]*u+y[j]*m[j]*u*u+m[j]*m[j]*u*u*u/3
@njit(cache=True)
def maximum(x,m,y,P,w,start,end):
 if end<start:return -1.,0.
 left=start;best=-1.;witness=start
 j=np.searchsorted(x,left,side='right')-1;k=np.searchsorted(x,left+w,side='right')-1
 while True:
  a=left-x[j];b=left+w-x[k]
  value=(P[k]+y[k]*y[k]*b+y[k]*m[k]*b*b+m[k]*m[k]*b*b*b/3)-(P[j]+y[j]*y[j]*a+y[j]*m[j]*a*a+m[j]*m[j]*a*a*a/3)
  if value>best:best=value;witness=left
  if left>=end:break
  while j+1<len(x) and x[j+1]<=left:j+=1
  while k+1<len(x) and x[k+1]<=left+w:k+=1
  right=end
  if j+1<len(x):right=min(right,x[j+1])
  if k+1<len(x):right=min(right,x[k+1]-w)
  if right<=left:
   left=np.nextafter(left,np.inf);continue
  mid=(left+right)/2
  v0=y[j]+m[j]*(mid-x[j]);v1=y[k]+m[k]*(mid+w-x[k])
  for sign in [-1,1]:
   den=m[k]+sign*m[j]
   if abs(den)>1e-12:
    q=mid-(v1+sign*v0)/den
    if q>left and q<right:
     a=q-x[j];b=q+w-x[k]
     value=(P[k]+y[k]*y[k]*b+y[k]*m[k]*b*b+m[k]*m[k]*b*b*b/3)-(P[j]+y[j]*y[j]*a+y[j]*m[j]*a*a+m[j]*m[j]*a*a*a/3)
     if value>best:best=value;witness=q
  left=right
 return max(0.,best),witness

def model(t,v,lo,hi):
 x0=np.r_[lo,hi,t];ds0=np.r_[0.,0.,v];order=np.argsort(x0);x0=x0[order];ds0=ds0[order];x,k=np.unique(x0,return_index=True);m=np.cumsum(np.add.reduceat(ds0,k));y=np.r_[0.,np.cumsum(m[:-1]*np.diff(x))];d=np.diff(x);P=np.r_[0.,np.cumsum(y[:-1]**2*d+y[:-1]*m[:-1]*d*d+m[:-1]**2*d**3/3)]
 def run(w,start,end):return maximum(x,m,y,P,w,start,end)
 return run
