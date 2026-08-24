"""Refit-per-surrogate permutation test.

For the real driver and for EVERY phase-randomised surrogate the entire
procedure is repeated: the closure coefficients (kappa, lambda, D_c[, ell])
are fitted on the training window, the memoryless baseline is fitted on the
same window with the same driver, and the skill is evaluated out of sample on
the disjoint test window.  The null distribution is therefore over fully
refitted models, not over one fixed model evaluated on scrambled input.

Primary endpoint: Delta C_in, first differences of the inner-region mean
connectivity.  Primary baseline: the model's own memoryless limit,
C*(r,t) = kappa G/(kappa G + lambda chi), aggregated over the same region.
"""
import sys, json, time
import numpy as np
from scipy.optimize import minimize
from core import load, Model, phase_rand
from fastcore import integrate_inner

T0    = float(sys.argv[1]) if len(sys.argv) > 1 else 7900.0
RCUT  = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
FRAC  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
NSUR  = int(sys.argv[4])   if len(sys.argv) > 4 else 200
SEED  = int(sys.argv[5])   if len(sys.argv) > 5 else 2026
OUT   = sys.argv[6]        if len(sys.argv) > 6 else None
MODEL = sys.argv[7]        if len(sys.argv) > 7 else "both"
NSTART, MAXITER = 8, 250

d = load(T0, RCUT); M = Model(d)
t, r, C, chi, G = d['t'], d['r'], d['C'], d['chi'], d['G']
n = M.n; imask = M.inner.astype(np.bool_); Cin = M.Cin; Gi = G[imask]
i0 = int(len(t)*FRAC); tr, te = slice(0, i0), slice(i0, len(t))
t_tr, t_te = t[tr].copy(), t[te].copy()
d_tr, d_te = np.diff(Cin[tr]), np.diff(Cin[te])
C0 = C[0].copy(); tg = t.copy()
lo, up, mid = M.lo, M.up, M.mid
e_zero = float(np.mean(d_te**2))

def wt(ell):
    return np.ones(n) if ell is None else np.exp(-(r - r[0])/ell)

def series(kap, lam, Dc, ell, tev, Cstart, cv):
    return integrate_inner(kap, lam, Dc, wt(ell), tev, Cstart, tg, cv,
                           G, lo, up, mid, 10.0, imask)

def fit(localised, cv, rng):
    def loss(p):
        kap, lam, Dc = np.exp(p[:3])
        ell = float(np.exp(p[3])) if localised else None
        if not (1e-3 < kap < 50 and 1e-3 < lam < 50 and 1e-6 < Dc < 5): return 1e9
        if localised and not (0.5 < ell < 40): return 1e9
        s, _ = series(kap, lam, Dc, ell, t_tr, C0, cv)
        return float(np.mean((np.diff(s) - d_tr)**2))
    best, bp = np.inf, None
    for _ in range(NSTART):
        p0 = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2),
                       rng.uniform(-6, 0), rng.uniform(0.5, 3)])
        res = minimize(loss, p0, method="Nelder-Mead",
                       options=dict(maxiter=MAXITER, xatol=1e-2, fatol=1e-12))
        if res.fun < best: best, bp = res.fun, res.x
    kap, lam, Dc = np.exp(bp[:3])
    return float(kap), float(lam), float(Dc), (float(np.exp(bp[3])) if localised else None), float(best)

def memoryless(cv, rng, nstart=6):
    """Memoryless limit of Eq. (2): D_c -> 0 and dC/dt -> 0 give
    C*(r,t) = kappa G/(kappa G + lambda chi), whose inner-region mean depends on
    the coefficients only through s = lambda/kappa.  The primary form carries an
    amplitude A as well, so that it has two free constants like Eq. (5); the
    restricted A = 1 form is recorded alongside it."""
    def f(s_, x):
        return np.mean(Gi[None, :]/(Gi[None, :] + s_*x[:, None]), axis=1)
    def L2(p):
        if abs(p[0]) > 20: return 1e9
        return float(np.mean((p[1]*np.diff(f(np.exp(p[0]), cv[tr])) - d_tr)**2))
    def L1(p):
        if abs(p[0]) > 20: return 1e9
        return float(np.mean((np.diff(f(np.exp(p[0]), cv[tr])) - d_tr)**2))
    b2 = min((minimize(L2, np.r_[rng.uniform(-4, 4), rng.uniform(0.2, 3)],
                       method="Nelder-Mead",
                       options=dict(maxiter=1500, xatol=1e-8, fatol=1e-16))
              for _ in range(nstart)), key=lambda z: z.fun)
    b1 = min((minimize(L1, rng.uniform(-4, 4, 1), method="Nelder-Mead",
                       options=dict(maxiter=1000, xatol=1e-8, fatol=1e-16))
              for _ in range(nstart)), key=lambda z: z.fun)
    s2, A2 = float(np.exp(b2.x[0])), float(b2.x[1])
    s1 = float(np.exp(b1.x[0]))
    e2 = float(np.mean((A2*np.diff(f(s2, cv[te])) - d_te)**2))
    e1 = float(np.mean((np.diff(f(s1, cv[te])) - d_te)**2))
    return e2, e1, [s2, A2, s1]

def full(cv_raw, localised, rng):
    cv = (cv_raw/cv_raw[tr].mean()).astype(float)
    kap, lam, Dc, ell, L = fit(localised, cv, rng)
    s1, Cend = series(kap, lam, Dc, ell, t_tr, C0, cv)
    s2, _    = series(kap, lam, Dc, ell, t_te, Cend, cv)
    e_mod = float(np.mean((np.diff(s2) - d_te)**2))
    e_mem, e_mem1, mem_p = memoryless(cv, rng)
    q = np.polyfit(cv[tr], Cin[tr], 1)
    e_lin = float(np.mean((np.diff(np.polyval(q, cv[te])) - d_te)**2))
    return dict(kappa=kap, lam=lam, D_c=Dc, ell=ell, train_loss=L,
                skill=float(1 - e_mod/e_mem),
                skill_mem1=float(1 - e_mod/e_mem1),
                skill_linear=float(1 - e_mod/e_lin),
                skill_clim=float(1 - e_mod/e_zero),
                mem_vs_clim=float(1 - e_mem/e_zero),
                e_mod=e_mod, e_mem=e_mem, e_mem1=e_mem1, e_lin=e_lin, mem_coef=mem_p)

res = {"T0": T0, "rcut": RCUT, "frac": FRAC, "nsur": NSUR, "seed": SEED,
       "nsamp": int(len(t)), "t0": float(t[0]), "t1": float(t[-1]),
       "nstart": NSTART, "maxiter": MAXITER, "e_zero": e_zero}
models = [(False, "scalar"), (True, "localised")]
if MODEL != "both": models = [m for m in models if m[1] == MODEL]
t_start = time.time()
for localised, nm in models:
    real = full(chi, localised, np.random.default_rng(SEED + 1))
    rec = []
    for j in range(NSUR):
        sur = phase_rand(chi, np.random.default_rng(SEED + 1000 + j))
        rec.append(full(sur, localised, np.random.default_rng(SEED + 500000 + j)))
        if (j+1) % 25 == 0:
            print(f"  {nm}: {j+1}/{NSUR}  ({time.time()-t_start:.0f}s)", flush=True)
    null = np.array([x["skill"] for x in rec])
    nlin = np.array([x["skill_linear"] for x in rec])
    nm1  = np.array([x["skill_mem1"] for x in rec])
    p     = float((np.sum(null >= real["skill"]) + 1)/(len(null) + 1))
    p_lin = float((np.sum(nlin >= real["skill_linear"]) + 1)/(len(nlin) + 1))
    p_m1  = float((np.sum(nm1 >= real["skill_mem1"]) + 1)/(len(nm1) + 1))
    res[nm] = dict(real=real, p=p, p_linear=p_lin, p_mem1=p_m1,
                   null=null.tolist(), null_linear=nlin.tolist(), null_mem1=nm1.tolist(),
                   null_median=float(np.median(null)),
                   null_p95=float(np.percentile(null, 95)),
                   null_max=float(null.max()))
    print(f"{nm:10s} real S = {real['skill']:+.3f} (linear-baseline {real['skill_linear']:+.3f}, "
          f"clim {real['skill_clim']:+.3f})   refit null: median {np.median(null):+.3f}, "
          f"95th {np.percentile(null,95):+.3f}, max {null.max():+.3f}   "
          f"p = {p:.4f}  (p_lin = {p_lin:.4f})", flush=True)
    if OUT: json.dump(res, open(OUT, "w"), indent=1)
print("total %.0fs" % (time.time()-t_start))
