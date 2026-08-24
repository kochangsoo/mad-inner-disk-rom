#!/usr/bin/env python3
"""
driven_test.py  --  Sect. 7 protocol, executable.

Drives the connectivity ROM with the MEASURED horizon flux chi(t) = phi_BH(t),
fits (kappa, lambda, D_c, C_c) once on a training window, and predicts
R_d,eff(t) on a disjoint held-out window.  Reports skill against two baselines:

  loc   the analytic local limit  R_d = a - ell*ln(chi)   (memoryless, 2 params)
  clim  the training-window mean of R_d,eff               (constant)

Pass/fail criteria are those pre-registered in Sect. 7.3.

INPUT
    <run>_rom.npz produced by reduce_run.py, containing at minimum
        time   (Nt,)      simulation time in GM/c^3
        radius (Nr,)      radial grid
        C      (Nt,Nr)    connectivity field from the density-threshold map
        chi    (Nt,)      normalised horizon magnetic flux
    Optionally G(r); if absent it is computed from <ln rho>_t if `rho` is
    present, else the exponential profile of Sect. 4 is used (and flagged).

USAGE
    python driven_test.py --rom a0p9375_rom.npz
    python driven_test.py --rom a0p9375_rom.npz --tskip 1500 --restarts 12
    python driven_test.py --rom a0p9375_rom.npz --rho-th-scan     # Sect. 7.4

OUTPUT
    driven_test_<run>.json     all numbers
    driven_test_<run>.png      training / test trajectories and residuals
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import minimize

RNG = np.random.default_rng(20260821)


# ----------------------------------------------------------------- I/O

def load_rom(path):
    d = np.load(path)
    need = ("time", "radius", "C", "chi")
    missing = [k for k in need if k not in d]
    if missing:
        sys.exit(f"{path}: missing {missing}. Re-run reduce_run.py.")
    out = {k: np.asarray(d[k], float) for k in need}
    out["G"] = np.asarray(d["G"], float) if "G" in d else None
    out["rho"] = np.asarray(d["rho"], float) if "rho" in d else None
    return out


def effective_radius(C, r):
    """Eq. (map-Rd): connectivity-weighted effective radius."""
    num = np.trapezoid(r[None, :] * C, r, axis=1)
    den = np.trapezoid(C, r, axis=1)
    return np.where(den > 0, num / np.maximum(den, 1e-30), r[0])


def gradient_profile(rom, r):
    """Eq. (map-G): time-averaged radial density gradient, else fallback."""
    if rom["G"] is not None:
        return np.abs(rom["G"]), "measured (supplied)"
    if rom["rho"] is not None:
        lr = np.log(np.maximum(rom["rho"], 1e-30)).mean(axis=0)
        return np.abs(np.gradient(lr, r)), "measured (from rho)"
    return 5.0 * np.exp(-(r - r[0]) / 8.0), "FALLBACK exponential -- REPORT THIS"


# ----------------------------------------------------------- ROM solver

class ROM:
    """Crank-Nicolson integrator for the connectivity PDE.

    The right-hand side is linear in C,
        dC/dt = [ D_c L - diag(kappa*G + lambda*chi(t)) ] C + kappa*G,
    with L the cylindrical Laplacian under Neumann conditions at both ends, so a
    semi-implicit step is unconditionally stable and reduces to one tridiagonal
    solve.  This is what makes the parameter fit of Sect. 7.1 affordable: an
    explicit scheme is limited to dt < dr^2/(2 D_c) and costs ~30x more.
    """

    def __init__(self, r, G, t, chi, dt=None):
        self.r, self.G = r, G
        self.dr = dr = r[1] - r[0]
        self.chi_t, self.chi_v = t, chi
        self.dt = dt if dt is not None else max(0.05, float(np.median(np.diff(t))) / 4.0)
        n = len(r)
        rp, rm = r + dr / 2, r - dr / 2
        lo = rm / (r * dr * dr)                 # coefficient on C_{i-1}
        up = rp / (r * dr * dr)                 # coefficient on C_{i+1}
        mid = -(lo + up)
        # Neumann (zero-gradient) at both ends, imposed by the mirror ghost cells
        # C_{-1} = C_1 and C_{N} = C_{N-2}.  This is second-order accurate and is
        # the discretisation used throughout Sects. 4-6; a first-order ghost
        # (C_{-1} = C_0) changes C by O(1) in the two innermost zones.
        up[0] += lo[0]
        lo[0] = 0.0
        lo[-1] += up[-1]
        up[-1] = 0.0
        self.L_lo, self.L_mid, self.L_up = lo, mid, up
        self.n = n

    def chi(self, t):
        return float(np.interp(t, self.chi_t, self.chi_v))

    def _bands(self, kap, lam, Dc, t, sign, h):
        """Assemble (I + sign*h/2*A(t)) in scipy solve_banded (l=1,u=1) layout."""
        f = sign * h / 2.0
        diag = 1.0 + f * (Dc * self.L_mid - (kap * self.G + lam * self.chi(t)))
        lower = f * Dc * self.L_lo
        upper = f * Dc * self.L_up
        ab = np.zeros((3, self.n))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        return ab, lower, diag, upper

    def _apply(self, lower, diag, upper, C):
        out = diag * C
        out[1:] += lower[1:] * C[:-1]
        out[:-1] += upper[:-1] * C[1:]
        return out

    def integrate(self, theta, t_eval, C0=None):
        kap, lam, Dc = theta
        if C0 is None:
            C0 = kap * self.G / (kap * self.G + lam * self.chi(t_eval[0]) + 1e-30)
        h = self.dt
        b = kap * self.G
        t = float(t_eval[0])
        C = np.clip(np.asarray(C0, float), 0.0, 1.0)
        out = np.empty((len(t_eval), self.n))
        out[0] = C
        for k in range(1, len(t_eval)):
            t_target = float(t_eval[k])
            nsub = max(1, int(np.ceil((t_target - t) / h)))
            hh = (t_target - t) / nsub
            for _ in range(nsub):
                _, lo_e, di_e, up_e = self._bands(kap, lam, Dc, t, +1.0, hh)
                rhs = self._apply(lo_e, di_e, up_e, C) + hh * b
                ab, *_ = self._bands(kap, lam, Dc, t + hh, -1.0, hh)
                C = solve_banded((1, 1), ab, rhs)
                if not np.all(np.isfinite(C)):
                    return None
                C = np.clip(C, 0.0, 1.0)
                t += hh
            out[k] = C
        return out


# ------------------------------------------------------------- fitting

# C_c does NOT enter the connectivity-weighted radius of Eq. (map-Rd), so it is
# not identifiable under this mapping and is excluded from the fit.  The effective
# number of free coefficients is therefore three, not four.
BOUNDS = [(1e-3, 50.0), (1e-3, 50.0), (1e-6, 5.0)]


def fit(rom_solver, t_tr, R_tr, restarts, seed_scale):
    """Minimise MSE of R_d against measured R_d,eff on the TRAINING window only."""
    def loss(p):
        theta = np.exp(p[:3]).tolist()
        for v, (lo, hi) in zip(theta, BOUNDS):
            if not (lo <= v <= hi):
                return 1e6
        C = rom_solver.integrate(theta, t_tr)
        if C is None or not np.all(np.isfinite(C)):
            return 1e6
        Rd = effective_radius(C, rom_solver.r) * seed_scale
        return float(np.mean((Rd - R_tr) ** 2))

    best, best_p = np.inf, None
    for k in range(restarts):
        p0 = np.array([RNG.uniform(-2, 2), RNG.uniform(-2, 2), RNG.uniform(-8, 0)])
        res = minimize(loss, p0, method="Nelder-Mead",
                       options=dict(maxiter=600, xatol=1e-3, fatol=1e-7))
        if res.fun < best:
            best, best_p = res.fun, res.x
    theta = np.exp(best_p[:3]).tolist()
    return theta, best


def fit_local_limit(chi_tr, R_tr):
    """Eq. (memoryless): R_d = a - ell*ln(chi).  Linear in (a, ell)."""
    X = np.column_stack([np.ones_like(chi_tr), -np.log(np.maximum(chi_tr, 1e-12))])
    coef, *_ = np.linalg.lstsq(X, R_tr, rcond=None)
    return coef                                            # (a, ell)


def identifiability(rom_solver, t_tr, R_tr, theta, mse0):
    """Fractional rise in training MSE when each coefficient is scaled by 0.5 and 2,
    the others held at the optimum.  A coefficient that barely moves the loss is not
    constrained by the data and must not be reported as a measurement."""
    names = ("kappa", "lam", "D_c")
    out = {}
    for i, nm in enumerate(names):
        worst = 0.0
        for f in (0.5, 2.0):
            th = list(theta)
            th[i] = float(np.clip(th[i] * f, *BOUNDS[i]))
            C = rom_solver.integrate(th, t_tr)
            if C is None:
                continue
            m = float(np.mean((effective_radius(C, rom_solver.r) - R_tr) ** 2))
            worst = max(worst, (m - mse0) / max(mse0, 1e-30))
        out[nm] = worst
    return out


# --------------------------------------------------------------- skill

def skill(pred, base, truth):
    num = np.mean((pred - truth) ** 2)
    den = np.mean((base - truth) ** 2)
    return 1.0 - num / den if den > 0 else np.nan


def block_bootstrap_skill(pred, base, truth, n_boot=2000):
    """Moving-block bootstrap; block length = integral autocorrelation time."""
    x = truth - truth.mean()
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    ac /= ac[0]
    stop = np.argmax(ac < 0.05) if np.any(ac < 0.05) else len(ac) // 4
    L = max(2, int(stop))
    n = len(truth)
    nb = int(np.ceil(n / L))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = RNG.integers(0, max(1, n - L), size=nb)
        idx = np.concatenate([np.arange(s, s + L) for s in starts])[:n]
        idx = idx[idx < n]
        out[b] = skill(pred[idx], base[idx], truth[idx])
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), L


def lag_of_peak(pred, truth, dt):
    a = pred - pred.mean()
    b = truth - truth.mean()
    if a.std() == 0 or b.std() == 0:
        return np.nan
    c = np.correlate(a, b, "full") / (len(a) * a.std() * b.std())
    return float((np.argmax(c) - (len(a) - 1)) * dt)


def phase_randomise(x):
    """Surrogate with the same power spectrum, randomised phases."""
    F = np.fft.rfft(x - x.mean())
    ph = RNG.uniform(0, 2 * np.pi, len(F))
    ph[0] = 0
    if len(x) % 2 == 0:
        ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x)) + x.mean()


# ---------------------------------------------------------------- main

def run(args):
    rom = load_rom(args.rom)
    t, r, C_sim, chi = rom["time"], rom["radius"], rom["C"], rom["chi"]

    ok = np.isfinite(chi) & np.isfinite(C_sim).all(axis=1)
    if args.tskip is not None:
        ok &= t >= args.tskip
    t, chi, C_sim = t[ok], chi[ok], C_sim[ok]
    if len(t) < 100:
        sys.exit("fewer than 100 usable samples after masking; check --tskip")

    R_meas = effective_radius(C_sim, r)
    G, G_src = gradient_profile(rom, r)

    # --- split ONCE, before any fitting
    isplit = len(t) // 2
    t_tr, t_te = t[:isplit], t[isplit:]
    R_tr, R_te = R_meas[:isplit], R_meas[isplit:]
    chi_tr, chi_te = chi[:isplit], chi[isplit:]

    # --- drive: normalise chi by its TRAINING mean only
    scale = chi_tr.mean()
    chi_n = chi / scale
    solver = ROM(r, G, t, chi_n, dt=args.dt)

    # R_d from the ROM lives on the same radial grid, so no rescaling is needed;
    # seed_scale is kept as an explicit 1.0 to make that assumption visible.
    seed_scale = 1.0

    theta, mse_tr = fit(solver, t_tr, R_tr, args.restarts, seed_scale)
    ident = identifiability(solver, t_tr, R_tr, theta, mse_tr)

    # --- predict on the held-out window, coefficients frozen
    C_tr = solver.integrate(theta, t_tr)
    C_te = solver.integrate(theta, t_te, C0=C_tr[-1])
    if C_te is None:
        sys.exit("integration failed on the test window")
    R_pde = effective_radius(C_te, r) * seed_scale

    # --- baselines fitted on the training window only
    a, ell = fit_local_limit(chi_n[:isplit], R_tr)
    R_loc = a - ell * np.log(np.maximum(chi_n[isplit:], 1e-12))
    R_clim = np.full_like(R_te, R_tr.mean())

    S_loc = skill(R_pde, R_loc, R_te)
    S_clim = skill(R_pde, R_clim, R_te)
    lo, hi, L = block_bootstrap_skill(R_pde, R_loc, R_te)
    dt = float(np.median(np.diff(t)))
    lag = lag_of_peak(R_pde, R_te, dt)

    # --- surrogate control
    solver_s = ROM(r, G, t, phase_randomise(chi_n), dt=args.dt)
    C_s = solver_s.integrate(theta, t_te)
    S_sur = skill(effective_radius(C_s, r) * seed_scale, R_loc, R_te) \
        if C_s is not None else np.nan

    verdict = ("supported" if (lo > 0 and abs(lag) <= dt)
               else "uninformative" if S_clim <= 0
               else "not supported")

    out = dict(
        rom_file=os.path.basename(args.rom), n_samples=int(len(t)),
        t_range=[float(t[0]), float(t[-1])], dt=dt,
        split_time=float(t[isplit]), G_source=G_src,
        theta=dict(kappa=theta[0], lam=theta[1], D_c=theta[2]),
        identifiability_dMSE_over_MSE=ident,
        train_mse=float(mse_tr),
        local_limit=dict(a=float(a), ell=float(ell)),
        skill_vs_local=float(S_loc), skill_ci95=[lo, hi], block_len=int(L),
        skill_vs_climatology=float(S_clim),
        skill_surrogate=float(S_sur),
        xcorr_lag=lag, verdict=verdict,
    )

    tag = os.path.basename(args.rom).replace("_rom.npz", "")
    with open(f"driven_test_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print()
    print("  Sect. 7.3 criteria")
    print(f"    supported      : skill_vs_local CI lower bound > 0  and |lag| <= dt")
    print(f"                     -> CI = [{lo:.3f}, {hi:.3f}], lag = {lag:.2f}, dt = {dt:.2f}")
    print(f"    not supported  : skill_vs_local <= 0        -> {S_loc:.3f}")
    print(f"    uninformative  : skill_vs_climatology <= 0  -> {S_clim:.3f}")
    print(f"    surrogate control (must be << skill_vs_local): {S_sur:.3f}")
    print("  identifiability (fractional rise in training MSE under x0.5 / x2):")
    for k, v in ident.items():
        flag = "  <- NOT constrained by the data" if v < 0.05 else ""
        print(f"    {k:8s} {v:10.3f}{flag}")
    print(f"    VERDICT: {verdict}")

    if args.plot:
        make_plot(tag, t_tr, R_tr, t_te, R_te, R_pde, R_loc, R_clim, chi_n, t, isplit)


def make_plot(tag, t_tr, R_tr, t_te, R_te, R_pde, R_loc, R_clim, chi_n, t, isplit):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                           gridspec_kw=dict(height_ratios=[2, 1], hspace=0.08))
    ax[0].plot(t_tr, R_tr, color="#555", lw=1.2, label=r"measured $R_{d,\rm eff}$ (train)")
    ax[0].plot(t_te, R_te, color="#111", lw=1.4, label=r"measured $R_{d,\rm eff}$ (test)")
    ax[0].plot(t_te, R_pde, color="#2E7D4F", lw=1.6, label="ROM prediction")
    ax[0].plot(t_te, R_loc, color="#8B3A8B", lw=1.2, ls="--", label="memoryless local limit")
    ax[0].plot(t_te, R_clim, color="#BBB", lw=1.0, ls=":", label="training mean")
    ax[0].axvline(t[isplit], color="#C1440E", lw=1.0)
    ax[0].set_ylabel(r"$R_d\ [GM/c^2]$")
    ax[0].legend(frameon=False, fontsize=8, ncol=2)
    ax[1].plot(t, chi_n, color="#999", lw=0.9)
    ax[1].axvline(t[isplit], color="#C1440E", lw=1.0)
    ax[1].set_ylabel(r"$\chi=\varphi_{\rm BH}$")
    ax[1].set_xlabel(r"$t\ [GM/c^3]$")
    for a_ in ax:
        a_.grid(alpha=0.25, lw=0.5)
    plt.savefig(f"driven_test_{tag}.png", dpi=170, bbox_inches="tight")
    print(f"\n  wrote driven_test_{tag}.png")


def selftest():
    """Generate data with this solver at known coefficients, then verify that the
    protocol recovers them and returns 'supported'.  Any change to the solver, the
    fit, or the skill definition should be re-checked against this."""
    rin, rout, Nr = 6.0, 50.0, 120
    r = np.linspace(rin, rout, Nr)
    G = 5.0 * np.exp(-(r - rin) / 8.0)
    t = np.arange(0, 4000, 4.0)
    rng = np.random.default_rng(7)
    period = 380.0
    ph = (t % period) / period
    chi = np.where(ph < 0.85, 0.3 + 1.2 * (1 - np.exp(-ph * 3.5)),
                   0.3 + 1.2 * (1 - np.exp(-0.85 * 3.5)) * np.exp(-(ph - 0.85) * period / 12.))
    chi = chi * (1 + 0.05 * rng.standard_normal(len(t)))
    true = [1.0, 1.5, 0.05]
    C = ROM(r, G, t, chi, dt=0.5).integrate(true, t)
    np.savez_compressed("selftest_rom.npz", time=t, radius=r, C=C, chi=chi, G=G)
    print("wrote selftest_rom.npz  (true kappa=1.0, lambda=1.5, D_c=0.05;")
    print(" note lambda is rescaled by the training-window mean of chi when fitted)")
    print("now run:  python driven_test.py --rom selftest_rom.npz --tskip 400")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom", required=False, help="<run>_rom.npz from reduce_run.py")
    ap.add_argument("--tskip", type=float, default=None,
                    help="discard t < tskip (pre-saturation transient)")
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--dt", type=float, default=None, help="CN step; default = data cadence")
    ap.add_argument("--plot", action="store_true", default=True)
    ap.add_argument("--selftest", action="store_true",
                    help="write selftest_rom.npz with known coefficients and exit")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run(a)
