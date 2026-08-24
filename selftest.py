"""Self-test: generate C_in from the solver at known coefficients, then run the
Sect. 7.1 protocol on it and check that the coefficients and a high skill come back."""
import numpy as np
from scipy.optimize import minimize
from core import load, Model
from fastcore import integrate_inner
TRUE=dict(kappa=0.20, lam=2.00, D_c=0.05)
d=load(7900.,5.0); M=Model(d)
t,r,C,chi,G=d['t'],d['r'],d['C'],d['chi'],d['G']
n=M.n; im=M.inner.astype(np.bool_); Gi=G[M.inner]; tg=t.copy()
cvfull=(chi/chi[:len(t)//2].mean()).astype(float)
w=np.ones(n)
syn,_=integrate_inner(TRUE['kappa'],TRUE['lam'],TRUE['D_c'],w,t.copy(),C[0].copy(),tg,cvfull,
                      G,M.lo,M.up,M.mid,10.,im)
Cin=syn
i0=len(t)//2; tr,te=slice(0,i0),slice(i0,len(t))
d_tr,d_te=np.diff(Cin[tr]),np.diff(Cin[te])
cv=(chi/chi[tr].mean()).astype(float)
def ser(k,l,D,tev,C0): return integrate_inner(k,l,D,w,tev,C0,tg,cv,G,M.lo,M.up,M.mid,10.,im)
def loss(p):
    k,l,D=np.exp(p)
    if not(1e-3<k<50 and 1e-3<l<50 and 1e-6<D<5): return 1e9
    s,_=ser(k,l,D,t[tr].copy(),C[0].copy()); return float(np.mean((np.diff(s)-d_tr)**2))
rng=np.random.default_rng(11); best,bp=np.inf,None
for _ in range(8):
    res=minimize(loss,rng.uniform(-3,1,3),method="Nelder-Mead",options=dict(maxiter=400,xatol=1e-4,fatol=1e-18))
    if res.fun<best: best,bp=res.fun,res.x
k,l,D=np.exp(bp)
s1,Ce=ser(k,l,D,t[tr].copy(),C[0].copy()); s2,_=ser(k,l,D,t[te].copy(),Ce)
e_mod=np.mean((np.diff(s2)-d_te)**2)
def f(s_,x): return np.mean(Gi[None,:]/(Gi[None,:]+s_*x[:,None]),axis=1)
def L2(p):
    if abs(p[0])>20: return 1e9
    return float(np.mean((p[1]*np.diff(f(np.exp(p[0]),cv[tr]))-d_tr)**2))
b2=min((minimize(L2,np.r_[rng.uniform(-4,4),rng.uniform(0.2,3)],method="Nelder-Mead",
        options=dict(maxiter=4000,xatol=1e-9,fatol=1e-18)) for _ in range(20)),key=lambda z:z.fun)
e_mem=np.mean((b2.x[1]*np.diff(f(np.exp(b2.x[0]),cv[te]))-d_te)**2)
print("true  kappa=%.4f lam=%.4f D_c=%.4f"%(TRUE['kappa'],TRUE['lam'],TRUE['D_c']))
print("fit   kappa=%.4f lam=%.4f D_c=%.6f   (err %.2f%%, %.2f%%)"%(k,l,D,100*abs(k/TRUE['kappa']-1),100*abs(l/TRUE['lam']-1)))
print("skill vs memoryless limit = %.4f   (e_mod %.3e, e_mem %.3e)"%(1-e_mod/e_mem,e_mod,e_mem))
for name,sc in (("D_c x0.5",0.5),("D_c x2",2.0)):
    print("   %-9s training error ratio %.4f"%(name,loss(np.log([k,l,D*sc]))/best))
for name,sc in (("kappa x0.5",0.5),("kappa x2",2.0)):
    print("   %-9s training error ratio %.4f"%(name,loss(np.log([k*sc,l,D]))/best))
for name,sc in (("lam x0.5",0.5),("lam x2",2.0)):
    print("   %-9s training error ratio %.4f"%(name,loss(np.log([k,l*sc,D]))/best))
