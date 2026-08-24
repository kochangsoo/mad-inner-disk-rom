"""Shared core for the refit-per-surrogate permutation test."""
import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize

def load(T0=7900.0, rcut=5.0):
    rom = np.load('mad_rom.npz'); mad = np.load('mad_mad.npz')
    t, r, C, chi, rho = rom['time'], rom['radius'], rom['C'], rom['chi'], rom['rho']
    mdot = mad['mdot']
    g = np.isfinite(chi) & np.isfinite(C).all(axis=1) & (mdot > 1e-4)
    t, C, chi, rho = t[g], C[g], chi[g], rho[g]
    s = t >= T0
    t, C, chi, rho = t[s], C[s], chi[s], rho[s]
    G = np.abs(np.gradient(np.log(np.maximum(rho, 1e-30)).mean(axis=0), r))
    return dict(t=t, r=r, C=C, chi=chi, G=G, inner=(r <= rcut))

class Model:
    def __init__(self, d):
        self.t = d['t']; self.r = r = d['r']; self.C = d['C']
        self.G = d['G']; self.inner = d['inner']
        self.n = n = len(r); dr = r[1] - r[0]
        lo = (r - dr/2)/(r*dr*dr); up = (r + dr/2)/(r*dr*dr)
        self.mid = -(lo + up)
        up = up.copy(); lo = lo.copy()
        up[0] += lo[0]; lo[0] = 0.0
        lo[-1] += up[-1]; up[-1] = 0.0
        self.lo, self.up = lo, up
        self.Cin = self.C[:, self.inner].mean(axis=1)

    def run(self, kap, lam, Dc, w, tev, C0, cv, h=10.0):
        t, G, n, lo, up, mid = self.t, self.G, self.n, self.lo, self.up, self.mid
        b = kap*G; tc = float(tev[0]); Cv = np.clip(C0.copy(), 0, 1)
        out = np.empty((len(tev), n)); out[0] = Cv
        for k in range(1, len(tev)):
            tt = float(tev[k]); ns = max(1, int(np.ceil((tt-tc)/h))); hh = (tt-tc)/ns
            for _ in range(ns):
                d0 = Dc*mid - (kap*G + lam*float(np.interp(tc, t, cv))*w)
                d1 = Dc*mid - (kap*G + lam*float(np.interp(tc+hh, t, cv))*w)
                rhs = Cv*(1 + hh/2*d0) + hh*b
                rhs[1:]  += hh/2*Dc*lo[1:]*Cv[:-1]
                rhs[:-1] += hh/2*Dc*up[:-1]*Cv[1:]
                ab = np.zeros((3, n)); ab[1] = 1 - hh/2*d1
                ab[0, 1:]  = -hh/2*Dc*up[:-1]
                ab[2, :-1] = -hh/2*Dc*lo[1:]
                Cv = np.clip(solve_banded((1, 1), ab, rhs), 0, 1); tc += hh
            out[k] = Cv
        return out

    def weight(self, ell):
        return np.ones(self.n) if ell is None else np.exp(-(self.r - self.r[0])/ell)

    def fit(self, localised, tr, cv, rng, nstart=3, maxiter=80):
        """Fit (kappa, lam, D_c[, ell]) on the training window for driver cv."""
        d_tr = np.diff(self.Cin[tr])
        def loss(p):
            kap, lam, Dc = np.exp(p[:3])
            ell = np.exp(p[3]) if localised else None
            if not (1e-3 < kap < 50 and 1e-3 < lam < 50 and 1e-6 < Dc < 5): return 1e9
            if localised and not (0.5 < ell < 40): return 1e9
            S = self.run(kap, lam, Dc, self.weight(ell), self.t[tr], self.C[0], cv)
            return float(np.mean((np.diff(S[:, self.inner].mean(axis=1)) - d_tr)**2))
        best, bp = np.inf, None
        for _ in range(nstart):
            p0 = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2),
                           rng.uniform(-6, 0), rng.uniform(0.5, 3)])
            res = minimize(loss, p0, method="Nelder-Mead",
                           options=dict(maxiter=maxiter, xatol=1e-2, fatol=1e-12))
            if res.fun < best: best, bp = res.fun, res.x
        kap, lam, Dc = np.exp(bp[:3])
        ell = float(np.exp(bp[3])) if localised else None
        return float(kap), float(lam), float(Dc), ell, float(best)

    def skill(self, kap, lam, Dc, ell, tr, te, cv):
        """Out-of-sample skill against the memoryless baseline refit on the same driver."""
        d_te = np.diff(self.Cin[te])
        q = np.polyfit(cv[tr], self.Cin[tr], 1)
        e_lin = np.mean((np.diff(np.polyval(q, cv[te])) - d_te)**2)
        w = self.weight(ell)
        S1 = self.run(kap, lam, Dc, w, self.t[tr], self.C[0], cv)
        S2 = self.run(kap, lam, Dc, w, self.t[te], S1[-1], cv)
        e_mod = np.mean((np.diff(S2[:, self.inner].mean(axis=1)) - d_te)**2)
        return float(1 - e_mod/e_lin)

def phase_rand(x, rng):
    F = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2*np.pi, len(F)); ph[0] = 0
    if len(x) % 2 == 0: ph[-1] = 0
    return np.fft.irfft(np.abs(F)*np.exp(1j*ph), n=len(x)) + x.mean()
