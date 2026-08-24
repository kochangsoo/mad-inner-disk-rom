"""Radial localisation: where the driver destroys connectivity, in the run and in the model.
Computed on the stationary window with first differences, as in Sect. 7.6."""
import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from core import load, Model
from fastcore import integrate
d=load(7900.,5.0); M=Model(d)
t,r,C,chi,G=d['t'],d['r'],d['C'],d['chi'],d['G']
n=len(r); dP=np.diff(chi)
def prof(X):
    return np.array([np.corrcoef(dP,np.diff(X[:,i]))[0,1] if np.diff(X[:,i]).std()>1e-14
                     else np.nan for i in range(n)])
def prof_dt(X):
    P=chi-np.polyval(np.polyfit(t,chi,1),t)
    Y=X-np.array([np.polyval(np.polyfit(t,X[:,i],1),t) for i in range(n)]).T
    return np.array([np.corrcoef(P,Y[:,i])[0,1] if Y[:,i].std()>1e-14 else np.nan for i in range(n)])
p_df=prof(C); p_dt=prof_dt(C)
P=json.load(open('results/out_primary_scalar.json'))['scalar']['real']
cv=(chi/chi[:len(t)//2].mean()).astype(float)
S=integrate(P['kappa'],P['lam'],P['D_c'],np.ones(n),t.copy(),C[0].copy(),t.copy(),cv,
            G,M.lo,M.up,M.mid,10.0,True)
p_mod=prof(S)
se=1/np.sqrt(len(dP)-3); band=np.tanh(2*se)
fig,ax=plt.subplots(figsize=(7.0,4.6))
ax.axhline(0,color='#bbb',lw=.8)
ax.axhspan(-band,band,color='#bbb',alpha=.25,lw=0)
ax.plot(r,p_df,'s-',ms=3.5,color='#C1440E',label='2D MAD run (first differences)')
ax.plot(r,p_dt,'o--',ms=3.5,color='#E08A5A',label='2D MAD run (linear detrend)')
ax.plot(r,p_mod,'-',lw=2,color='#2E7D4F',label=r'fitted closure, scalar $\chi(t)$')
ax.set_xlim(r.min(),30); ax.set_ylim(-1.05,0.45)
ax.set_xlabel(r'$r\ [GM/c^2]$')
ax.set_ylabel(r'corr$\,[\Delta\varphi_{\rm BH},\,\Delta C(r,t)]$')
j=np.nanargmax(np.where(np.abs(p_df)>band,r,-1))
ax.axvspan(r.min(),r[j],color='#C1440E',alpha=.07)
ax.text(0.5*(r.min()+r[j]),0.30,'response confined\nhere in the run',fontsize=8.5,
        ha='center',color='#8A3A18')
ax.text(20,-0.90,'the closure disrupts\nat all radii',fontsize=8.5,color='#2E7D4F',ha='center')
ax.text(28.5,band+0.02,r'$2\sigma$',fontsize=8,color='#777',ha='right',va='bottom')
ax.grid(alpha=.25,lw=.5); ax.legend(frameon=False,fontsize=8.5,loc='lower left')
ax.set_title('A scalar driver cannot localise the disruption',fontsize=11)
plt.tight_layout(); plt.savefig('fig_local.pdf')
plt.savefig('fig_local.png',dpi=300)
print("run: peak r=%.2f rho=%.3f ; |rho|>2sigma out to r=%.2f (2sigma band %.3f)"%(
    r[np.nanargmin(p_df)],np.nanmin(p_df),r[j],band))
print("model: range %.3f to %.3f"%(np.nanmin(p_mod),np.nanmax(p_mod)))
