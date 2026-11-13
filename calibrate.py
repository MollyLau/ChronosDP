from pathlib import Path
import numpy as np,json,itertools,sys
from numba import njit
from src.fast_energy import model
name=sys.argv[1];D=Path('data/processed')/name;O=Path('results/calibration')/name;O.mkdir(parents=True,exist_ok=True)
p=json.load(open('experiments/config.json'));ratios=np.array(p['w_over_T']);ws=sorted(set(t*r for t in p['T'] for r in ratios));files=sorted(D.glob('*.npz'));dates=[f.stem for f in files];streams=[(np.load(f)['times'],np.load(f)['weights']) for f in files]
@njit(cache=True)
def hard_max(A,ratios):
 n,L=A.shape;best=np.zeros(len(ratios));wit=np.zeros((len(ratios),3),np.int64)
 upper=np.zeros((n,len(ratios)))
 for i in range(n):
  for r in range(len(ratios)):
   length=int(2*ratios[r])
   for e in range(length,L):upper[i,r]=max(upper[i,r],A[i,e]-A[i,e-length])
 order=np.argsort(upper[:,-1])[::-1]
 for ii in range(n):
  i=order[ii]
  for jj in range(ii):
   j=order[jj]
   if not np.any(upper[i]+upper[j]>best):continue
   # A contains cumulative mass at half-T boundaries. Compare full-bin
   # coordinates, plus the earliest partial bin for fractional windows.
   for parity in range(2):
    pref=np.zeros(L)
    for e in range(2+parity,L,2):pref[e]=pref[e-2]+abs((A[i,e]-A[i,e-2])-(A[j,e]-A[j,e-2]))
    for r in range(len(ratios)):
     if upper[i,r]+upper[j,r]<=best[r]:continue
     k=int(ratios[r]);frac=ratios[r]>k
     for e in range(2*k+parity+(2 if frac and parity==0 else 0),L,2):
      d=pref[e]-pref[e-2*k]
      if frac:d+=abs((A[i,e-2*k]-A[i,e-2*k-1])-(A[j,e-2*k]-A[j,e-2*k-1]))
      if d>best[r]:best[r]=d;wit[r,0]=i;wit[r,1]=j;wit[r,2]=e
 return best,wit
hard=[]
if not (O/'daily_hard.json').exists():
 for T in p['T']:
  q=np.arange(0,86400+1e-8,T/2);A=[]
  for s,v in streams:A.append(np.r_[0,np.cumsum(v)][np.searchsorted(s,q,side='left')])
  A=np.array(A);values,witness=hard_max(A,ratios);bins=np.diff(A[:,::2],axis=1);scalar=float(np.ptp(bins,axis=0).max())
  for ratio,D,(i,j,e) in zip(ratios,values,witness):
   assert D+1e-6>=scalar,(name,T,ratio,D,scalar)
   hard.append(dict(T=T,w=T*ratio,L1=float(D),scalar_max=scalar,dates=[dates[i],dates[j]],end=float(q[e]),N=86400//T))
  print(name,'hard',T,flush=True)
 (O/'daily_hard.json').write_text(json.dumps(hard,indent=2))
out=[]
if (O/'daily_brownian.json').exists():out=json.load(open(O/'daily_brownian.json'))
for tau in p['tau']:
 if any(r['tau']==tau for r in out):continue
 def make(s,v):return model(np.r_[s,s+tau,s+2*tau],np.r_[v,-2*v,v]/tau**2,0.,86400.)
 # Nonnegative rates imply (r_i-r_j)^2 <= r_i^2+r_j^2.
 # This bound skips only pairs that cannot improve any current maximum.
 upper=np.array([[make(s,v)(w,0.,86400.-w)[0] for w in ws] for s,v in streams])
 best=np.zeros(len(ws));witness=[None]*len(ws)
 pairs=sorted(itertools.combinations(range(len(streams)),2),key=lambda ij:float((upper[ij[0]]+upper[ij[1]]).max()),reverse=True)
 examined=0
 for i,j in pairs:
  candidates=np.flatnonzero(upper[i]+upper[j]>best*(1-1e-12))
  if not len(candidates):continue
  si,vi=streams[i];sj,vj=streams[j];maximum=make(np.r_[si,sj],np.r_[vi,-vj]);examined+=1
  for k in candidates:
   value,start=maximum(ws[k],0.,86400.-ws[k])
   if value>best[k]:best[k]=value;witness[k]=[dates[i],dates[j],start,start+ws[k]]
 for k,w in enumerate(ws):out.append(dict(w=w,tau=tau,window_energy=float(best[k]),witness=witness[k],day_cover=int(np.ceil(86400/w)),day_energy_bound=float(np.ceil(86400/w)*best[k])))
 (O/'daily_brownian.json').write_text(json.dumps(out,indent=2));print(name,'Brownian',tau,'pairs evaluated',examined,'of',len(pairs),flush=True)
print('DAILY_CALIBRATION_COMPLETE',name,flush=True)
