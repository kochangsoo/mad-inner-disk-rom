"""Fig. 4 -- the Lambda_min scan, with the two saturated points marked.

Reproduces the scan of the earlier version and shows that the reported exponent
is set by two points at which R_d has collapsed onto the inner boundary.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

L    = np.array([5.2271, 3.4174, 2.5385, 2.0192, 1.4329, 1.1105, 0.7659, 0.5845])
RR   = np.array([7.583, 9.340, 10.460, 11.102, 12.298, 13.130, 36.158, 56.846])
Rmin = np.array([19.4, 15.9, 13.5, 11.6, 8.9, 6.8, 6.00, 6.00])
chi1 = np.array([0.5, 1, 1.5, 2, 3, 4, 6, 8])
rin  = 6.0
sat  = np.isclose(Rmin, rin)

a_all, b_all = np.polyfit(np.log(L), np.log(RR), 1)
a_uns, b_uns = np.polyfit(np.log(L[~sat]), np.log(RR[~sat]), 1)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
xx = np.linspace(L.min()*0.85, L.max()*1.15, 100)
ax.loglog(xx, np.exp(b_all)*xx**a_all, '--', color='#999', lw=1.3,
          label=rf'all 8 points:  $\alpha={a_all:+.2f}$')
ax.loglog(xx, np.exp(b_uns)*xx**a_uns, '-', color='#1F6FB4', lw=1.6,
          label=rf'unsaturated only:  $\alpha={a_uns:+.2f}$')
ax.loglog(L[~sat], RR[~sat], 'o', ms=8, color='#1F6FB4',
          label=r'$R_{\min}>r_{\rm in}$')
ax.loglog(L[sat], RR[sat], 'o', ms=9, mfc='none', mew=1.8, color='#C1440E',
          label=r'$R_{\min}=r_{\rm in}$ (saturated)')
for x, y, c in zip(L[sat], RR[sat], chi1[sat]):
    ax.annotate(rf'$\chi_1={c:.0f}$', (x, y), textcoords='offset points',
                xytext=(9, -3), fontsize=8.5, color='#C1440E')
ax.set_xlabel(r'$\Lambda_{\min}$')
ax.set_ylabel(r'$\tau_{\rm rec}/\tau_{\rm col}$')
ax.grid(alpha=.25, lw=.5, which='both')
ax.legend(frameon=False, fontsize=9, loc='upper right')
ax.set_title(r'Two saturated points steepen the exponent by $\sim\!2.5\times$',
             fontsize=10.5)
plt.tight_layout(); plt.savefig('fig4.pdf')
plt.savefig('fig4.png', dpi=300)
print(f'all 8   alpha = {a_all:+.3f}')
print(f'6 unsat alpha = {a_uns:+.3f}   ratio = {a_all/a_uns:.2f}')
