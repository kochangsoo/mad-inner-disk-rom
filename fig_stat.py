"""Fig: (a) the response saturates; (b) the refit-per-surrogate null."""
import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
rom=np.load('mad_rom.npz'); mad=np.load('mad_mad.npz')
t,r,C,chi=rom['time'],rom['radius'],rom['C'],rom['chi']; mdot=mad['mdot']
g=np.isfinite(chi)&np.isfinite(C).all(axis=1)&(mdot>1e-4)
t,C,chi=t[g],C[g],chi[g]
Cin=C[:,r<=5.].mean(axis=1)
q=len(t)//4; rows=[]
for j in range(4):
    s=slice(j*q,(j+1)*q); dP,dC=np.diff(chi[s]),np.diff(Cin[s]); nn=len(dP)
    rr=np.corrcoef(dP,dC)[0,1]; se=1/np.sqrt(nn-3)
    rows.append((0.5*(t[s][0]+t[s][-1]),rr,np.tanh(np.arctanh(rr)-se),
                 np.tanh(np.arctanh(rr)+se),np.arctanh(rr)*np.sqrt(nn-3),t[s][0],t[s][-1],nn))
rows=np.array(rows)
print("quarters:")
for a in rows: print("  t %.0f-%.0f n=%d rho=%.3f z=%.1f"%(a[5],a[6],a[7],a[1],a[4]))
D=json.load(open('results/out_primary_localised.json'))
d=D['localised']; null=np.array(d['null']); S=d['real']['skill']; p=d['p']
fig,ax=plt.subplots(1,2,figsize=(10.4,3.9))
a0=ax[0]
a0.errorbar(rows[:,0],rows[:,1],yerr=[rows[:,1]-rows[:,2],rows[:,3]-rows[:,1]],
            fmt='o-',color='#C1440E',ms=6,capsize=3,lw=1.4)
for a in rows:
    a0.annotate(r'$z=%.0f$'%a[4],(a[0],a[1]),textcoords='offset points',
                xytext=(0,10 if a[1]<-0.5 else -16),ha='center',fontsize=8,color='#8A3A18')
a0.axvspan(7900,t[-1],color='#2E7D4F',alpha=0.10)
a0.text(0.5*(7900+t[-1]),rows[:,1].max()-0.03,'stationary window',color='#2E7D4F',
        fontsize=9,ha='center')
a0.set_xlabel(r'$t\ [GM/c^{3}]$')
a0.set_ylabel(r'corr$[\Delta\varphi_{\rm BH},\ \Delta C_{\rm in}]$')
a0.set_title('(a) the response saturates',fontsize=11); a0.grid(alpha=.25,lw=.5)
a1=ax[1]
a1.hist(null,bins=30,color='#999',edgecolor='none')
a1.axvline(np.percentile(null,95),color='#555',ls=':',lw=1.2)
a1.text(np.percentile(null,95),a1.get_ylim()[1]*0.75,' 95th pct\n of null',fontsize=8,color='#555')
a1.axvline(S,color='#C1440E',lw=2.2)
a1.text(S,a1.get_ylim()[1]*0.55,'  measured driver\n  %.2f   $p=%.3f$'%(S,p),
        fontsize=9,color='#C1440E',ha='left')
a1.set_xlim(min(null.min()-0.02,-0.05),S*1.18)
a1.set_xlabel('out-of-sample skill vs the memoryless limit')
a1.set_ylabel('refitted surrogates (%d)'%len(null))
a1.set_title('(b) the skill requires the real driver',fontsize=11); a1.grid(alpha=.25,lw=.5)
fig.tight_layout(); fig.savefig('fig_stat.pdf',bbox_inches='tight')
fig.savefig('fig_stat.png',dpi=300,bbox_inches='tight')
print("S=%.4f p=%.4f null: med %.4f p95 %.4f max %.4f"%(S,p,np.median(null),np.percentile(null,95),null.max()))
