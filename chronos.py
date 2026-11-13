import numpy as np
from scipy.special import erf

def absolute_mean(b,sigma):
 if sigma==0:return np.abs(b)
 z=b/sigma;return sigma*np.sqrt(2/np.pi)*np.exp(-.5*z*z)+b*erf(z/np.sqrt(2))

def smooth_model(s,v,tau):
 s=np.asarray(s,float);v=np.asarray(v,float)
 if tau<=0:raise ValueError('tau must be positive')
 if len(s)==0:return lambda q:np.zeros_like(np.asarray(q),dtype=float)
 t=np.r_[s,s+tau,s+2*tau];dm=np.r_[v,-2*v,v]/tau**2;ix=np.argsort(t);t=t[ix];dm=dm[ix];x,k=np.unique(t,return_index=True);m=np.cumsum(np.add.reduceat(dm,k));r=np.r_[0,np.cumsum(m[:-1]*np.diff(x))];P=np.r_[0,np.cumsum(r[:-1]*np.diff(x)+.5*m[:-1]*np.diff(x)**2)]
 def A(q):
  j=np.searchsorted(x,q,side='right')-1;valid=j>=0;j=np.clip(j,0,len(x)-1);u=q-x[j];return np.where(valid,P[j]+r[j]*u+.5*m[j]*u*u,0)
 return A

def rho_from_epsilon(epsilon,delta):
    return (epsilon/(np.sqrt(np.log(1/delta)+epsilon)+np.sqrt(np.log(1/delta))))**2

def noise_scale(day_energy_bound,epsilon,delta):
    return np.sqrt(day_energy_bound/(2*rho_from_epsilon(epsilon,delta)))

def brownian_path(times,scale,seed=20261111):
    """One path at a fixed collection of endpoints; do not redraw per query."""
    times=np.asarray(times,float)
    if (times<0).any():raise ValueError('Negative endpoint')
    unique,inverse=np.unique(np.r_[0.,times],return_inverse=True)
    rng=np.random.default_rng(seed)
    values=np.r_[0.,np.cumsum(rng.normal(size=len(unique)-1)*np.sqrt(np.diff(unique)))]*scale
    return values[inverse[1:]]

def release_at_endpoints(event_times,weights,tau,endpoints,scale,seed=20261111):
    return smooth_model(event_times,weights,tau)(np.asarray(endpoints))+brownian_path(endpoints,scale,seed)
