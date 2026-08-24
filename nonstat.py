"""Control: the same protocol applied to the full accreting interval, which is not stationary."""
import numpy as np, json, sys
from scipy.optimize import minimize
from core import load, Model, phase_rand
from fastcore import integrate_inner
T0=float(sys.argv[1]); NS=int(sys.argv[2])
d=load(T0,5.0); M=Model(d)
t,r,C,chi,G=d['t'],d['r'],d['C'],d['chi'],d['G']
n=M.n; im=M.inner.astype(np.bool_); Cin=M.Cin; Gi=G[M.inner]; tg=t.copy()
print("window t=%.0f..%.0f  n=%d"%(t[0],t[-1],len(t)))
def build(frac):
    i0=int(len(t)*frac); return slice(0,i0),slice(i0,len(t))
def wt(e): return np.ones(n) if e is None else np.exp(-(r-r[0])/e)
def run_case(cv_raw,localised,frac,rng):
    tr,te=build(frac); d_tr,d_te=np.diff(Cin[tr]),np.diff(Cin[te])
    cv=(cv_raw/cv_raw[tr].mean()).astype(float)
    def ser(k,l,D,e,tev,C0):
        return integrate_inner(k,l,D,wt(e),tev,C0,tg,cv,G,M.lo,M.up,M.mid,10.,im)
    def loss(p):
        k,l,D=np.exp(p[:3]); e=float(np.exp(p[3])) if localised else None
        if not(1e-3<k<50 and 1e-3<l<50 and 1e-6<D<5): return 1e9
        if localised and not(0.5<e<40): return 1e9
        s,_=ser(k,l,D,e,t[tr].copy(),C[0].copy()); return float(np.mean((np.diff(s)-d_tr)**2))
    best,bp=np.inf,None
    for _ in range(8):
        res=minimize(loss,np.array([rng.uniform(-2,2),rng.uniform(-2,2),rng.uniform(-6,0),rng.uniform(.5,3)]),
                     method="Nelder-Mead",options=dict(maxiter=250,xatol=1e-2,fatol=1e-12))
        if res.fun<best: best,bp=res.fun,res.x
    k,l,D=np.exp(bp[:3]); e=float(np.exp(bp[3])) if localised else None
    s1,Ce=ser(k,l,D,e,t[tr].copy(),C[0].copy()); s2,_=ser(k,l,D,e,t[te].copy(),Ce)
    e_mod=np.mean((np.diff(s2)-d_te)**2)
    def f(s_,x): return np.mean(Gi[None,:]/(Gi[None,:]+s_*x[:,None]),axis=1)
    def L2(p):
        if abs(p[0])>20: return 1e9
        return float(np.mean((p[1]*np.diff(f(np.exp(p[0]),cv[tr]))-d_tr)**2))
    b2=min((minimize(L2,np.r_[rng.uniform(-4,4),rng.uniform(0.2,3)],method="Nelder-Mead",
            options=dict(maxiter=1500,xatol=1e-8,fatol=1e-16)) for _ in range(6)),key=lambda z:z.fun)
    e_mem=np.mean((b2.x[1]*np.diff(f(np.exp(b2.x[0]),cv[te]))-d_te)**2)
    return float(1-e_mod/e_mem)
out={}
for localised,nm in ((False,'scalar'),(True,'localised')):
    row=[]
    for frac in (0.4,0.5,0.6):
        row.append(run_case(chi,localised,frac,np.random.default_rng(77)))
    print("%-10s real skill at frac 0.4/0.5/0.6: %+.3f %+.3f %+.3f"%(nm,*row),flush=True)
    null=[run_case(phase_rand(chi,np.random.default_rng(900+j)),localised,0.5,np.random.default_rng(1900+j)) for j in range(NS)]
    null=np.array(null); p=(np.sum(null>=row[1])+1)/(NS+1)
    print("   surrogate null (N=%d): median %+.3f  max %+.3f  p=%.3f"%(NS,np.median(null),null.max(),p),flush=True)
    out[nm]=dict(splits=row,null=null.tolist(),p=float(p))
json.dump(out,open('out_nonstat_%d.json'%int(T0),'w'),indent=1)
