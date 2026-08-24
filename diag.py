"""Null-test diagnostics for the connectivity ROM (Fig. 3 of the revised manuscript).

(a) tau_rec/tau_col is the prescribed driver asymmetry, inverted, times O(1).
(b) The ratio is set by the recovery-threshold convention.
(c) The same equations and coefficients give opposite asymmetry for the two driver
    shapes; the MAD-shaped driver lands inside the measured 2D range.
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

rin, rout, Nr = 6.0, 50.0, 200
G0, ell, kap, lam, Dc, Cc = 5.0, 8.0, 1.0, 1.5, 0.05, 0.5
chi0, ton = 0.3, 50.0
r = np.linspace(rin, rout, Nr); dr = r[1]-r[0]
G = G0*np.exp(-(r-rin)/ell)

def lap(C):
    Cg = np.concatenate([[C[1]], C, [C[-2]]])
    rp, rm = (r+dr/2), (r-dr/2)
    return (rp*(Cg[2:]-Cg[1:-1]) - rm*(Cg[1:-1]-Cg[:-2]))/(r*dr*dr)

def Rd(C):
    idx = np.where(C >= Cc)[0]
    if len(idx) == 0: return rin
    i = idx[-1]
    if i == Nr-1: return r[-1]
    f = (C[i]-Cc)/(C[i]-C[i+1]) if C[i] != C[i+1] else 0.
    return r[i] + f*dr

def run(chif, T, dt=0.05):
    C = kap*G/(kap*G + lam*chif(0.)); ts=[0.]; Rs=[Rd(C)]; Xs=[chif(0.)]; t=0.
    while t < T:
        f = lambda t_, y: Dc*lap(y) + kap*G*(1-y) - lam*chif(t_)*y
        k1=f(t,C); k2=f(t+dt/2,C+dt/2*k1); k3=f(t+dt/2,C+dt/2*k2); k4=f(t+dt,C+dt*k3)
        C = C + dt/6*(k1+2*k2+2*k3+k4); t += dt
        ts.append(t); Rs.append(Rd(C)); Xs.append(chif(t))
    return np.array(ts), np.array(Rs), np.array(Xs)

def sawtooth(chi1, tau_up, tau_down):
    """Generic asymmetric driver: saturating rise over tau_up, exponential fall over
       tau_down.  tau_up << tau_down reproduces the fast-rise/slow-decay pulse used in
       the original manuscript; tau_up >> tau_down reproduces the MAD flux-accumulation
       / eruption sawtooth."""
    tpk = ton + 3.0*tau_up
    def f(t):
        if t < ton: return chi0
        if t <= tpk: return chi0 + chi1*(1-np.exp(-(t-ton)/tau_up))
        pk = chi0 + chi1*(1-np.exp(-(tpk-ton)/tau_up))
        return chi0 + (pk-chi0)*np.exp(-(t-tpk)/tau_down)
    return f, tpk

def measure(ts, Rs, frac=0.9):
    i0 = np.searchsorted(ts, ton); R0 = Rs[i0-1]
    imin = i0 + int(np.argmin(Rs[i0:])); Rmin = Rs[imin]
    tcol = ts[imin] - ton
    tgt = Rmin + frac*(R0-Rmin)
    j = imin + int(np.argmax(Rs[imin:] >= tgt))
    return tcol, ts[j]-ts[imin], ts[imin], ts[j], R0, Rmin

PAIRS = [(2.,80.),(2.,40.),(2.,20.),(5.,20.),(5.,10.),(5.,5.),
         (10.,5.),(20.,5.),(20.,2.),(40.,2.),(80.,2.),(160.,2.)]

fig = plt.figure(figsize=(13.6, 4.3))
gs  = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1], hspace=0.55, wspace=0.30)
axA = fig.add_subplot(gs[:, 0]); axB = fig.add_subplot(gs[:, 1])
axC = fig.add_subplot(gs[0, 2]); axD = fig.add_subplot(gs[1, 2])

# ---- (a)
x, y = [], []
for tu, td in PAIRS:
    f, tpk = sawtooth(1.0, tu, td)
    ts, Rs, _ = run(f, tpk + 15*td + 400.)
    tc, tr, *_ = measure(ts, Rs)
    x.append(td/tu); y.append(tr/tc)
axA.loglog(x, y, 'o', ms=7, color='#C1440E', zorder=3)
xx = np.array([min(x)*0.7, max(x)*1.4])
axA.loglog(xx, 1.1*xx, '--', color='#555', lw=1.2, label=r'$1.1\,\tau_{\rm down}/\tau_{\rm up}$')
axA.axhspan(0.22, 2.64, color='#C1440E', alpha=0.12, zorder=1)
axA.text(0.013, 0.42, '2D MAD\nmeasured', fontsize=7.5, color='#8A3A18')
axA.set_xlabel(r'prescribed driver asymmetry  $\tau_{\rm down}/\tau_{\rm up}$')
axA.set_ylabel(r'measured  $\tau_{\rm rec}/\tau_{\rm col}$')
axA.set_title('(a) The output is the input, inverted', fontsize=11)
axA.legend(frameon=False, fontsize=9, loc='upper left')

# ---- (b)  uses the ORIGINAL manuscript pulse at its fiducial values, so that the
#           90% entry reproduces the value quoted in the earlier version.
def pulse(chi1, tau_r, tau_d):
    def f(t):
        if t < ton: return chi0
        s = t - ton
        return chi0 + chi1*(1-np.exp(-s/tau_r))*np.exp(-s/tau_d)
    return f
ts, Rs, _ = run(pulse(1.0, 1., 8.), 800.)
fr = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
rv = [measure(ts, Rs, frac=q)[1]/measure(ts, Rs, frac=q)[0] for q in fr]
print('panel (b) values:', [round(v,2) for v in rv])
axB.plot(fr*100, rv, 's-', color='#1F6FB4', ms=6)
axB.set_xlabel('recovery threshold (% of pre-burst $R_d$)')
axB.set_ylabel(r'$\tau_{\rm rec}/\tau_{\rm col}$')
axB.axhline(9.34, ls=':', color='#888', lw=1)
axB.text(51, 9.7, 'value quoted in the earlier version', fontsize=7.5, color='#666')
axB.set_title('(b) A convention, not a measurement', fontsize=11)

# ---- (c) two drivers, same equations
for axx, (tu, td), name, col in [
        (axC, (2., 20.),  r'fast-rise driver  $\tau_{\rm up}/\tau_{\rm down}=0.1$',  '#8B3A8B'),
        (axD, (40., 2.),  r'slow-rise driver  $\tau_{\rm up}/\tau_{\rm down}=20$',   '#2E7D4F')]:
    f, tpk = sawtooth(1.0, tu, td)
    ts, Rs, Xs = run(f, tpk + 12*td + 260.)
    tc, tr, tmin, trec_end, R0, Rmin = measure(ts, Rs)
    m = ts >= ton - 30
    axx.plot(ts[m]-ton, Rs[m], color=col, lw=1.8, zorder=3)
    ax2 = axx.twinx()
    ax2.plot(ts[m]-ton, Xs[m], color='#999', lw=1.0, ls='--', zorder=2)
    ax2.set_yticks([]); ax2.set_ylim(0, 1.6)
    axx.axvspan(0, tc, color=col, alpha=0.13, zorder=1)
    axx.axvspan(tc, trec_end-ton, color='#BBB', alpha=0.30, zorder=1)
    axx.set_ylabel(r'$R_d$', fontsize=9)
    axx.tick_params(labelsize=8)
    axx.set_title(name+r':   $\tau_{\rm rec}/\tau_{\rm col}$'+f' = {tr/tc:.2f}',
                  fontsize=9.0, color=col)
axD.set_xlabel(r'$t - t_{\rm on}$  $[GM/c^3]$', fontsize=9)
axD.text(0.0, -0.42, r'(c) same equations, same coefficients; shaded: collapse | recovery, dashed grey: $\chi(t)$',
         transform=axD.transAxes, ha='left', fontsize=8, color='#555')

for a in (axA, axB, axC, axD): a.grid(alpha=0.25, lw=0.5, which='both')
plt.savefig('fig_null.pdf', bbox_inches='tight')
plt.savefig('fig_null.png', dpi=300, bbox_inches='tight')
print('ok')
