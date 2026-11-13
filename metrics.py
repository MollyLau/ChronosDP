import numpy as np,math
from numba import njit
@njit(cache=True)
def evaluate(bias,var,truth,on):
 out=np.zeros((len(var),7));n=len(bias);ng=0;np_=0;npg=0;nz=0;b2=0.;b2g=0.;c=math.sqrt(2/math.pi)
 for i in range(n):
  b=bias[i];b2+=b*b
  if on[i]:ng+=1;b2g+=b*b
  if truth[i]>0:
   np_+=1
   if on[i]:npg+=1
  else:nz+=1
  for e in range(len(var)):
   sig=math.sqrt(var[e])
   if sig==0:ae=abs(b)
   else:
    z=b/sig
    if abs(z)<.01:
     # Folded-normal analytic expansion. The omitted z^10 term is
     # bounded by 3e-25 of sigma here, below double precision.
     u=z*z;ae=sig*c*(1+u*(.5+u*(-1/24+u*(1/240-u/2688))))
    else:ae=sig*c*math.exp(-z*z/2)+b*math.erf(z/math.sqrt(2))
   if truth[i]>0:
    out[e,3]+=ae/truth[i]
    if on[i]:out[e,6]+=ae/truth[i]
   else:out[e,4]+=ae
 for e in range(len(var)):
  out[e,0]=b2/n;out[e,1]=var[e];out[e,2]=b2/n+var[e];out[e,3]=out[e,3]/np_ if np_ else np.nan;out[e,4]=out[e,4]/nz if nz else np.nan;out[e,5]=b2g/ng+var[e];out[e,6]=out[e,6]/npg if npg else np.nan
 return out
