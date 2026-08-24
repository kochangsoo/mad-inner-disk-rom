"""Numba-accelerated integrator, numerically equivalent to the scipy version."""
import numpy as np
from numba import njit
from scipy.optimize import minimize

@njit(cache=True, fastmath=False)
def _interp(x, xp, fp):
    n = len(xp)
    if x <= xp[0]: return fp[0]
    if x >= xp[n-1]: return fp[n-1]
    lo, hi = 0, n-1
    while hi - lo > 1:
        mid = (lo+hi)//2
        if xp[mid] <= x: lo = mid
        else: hi = mid
    w = (x - xp[lo])/(xp[hi]-xp[lo])
    return fp[lo]*(1.0-w) + fp[hi]*w

@njit(cache=True, fastmath=False)
def integrate(kap, lam, Dc, w, tev, C0, tgrid, cv, G, lo, up, mid, h, want_all):
    n = C0.shape[0]; m = tev.shape[0]
    out = np.empty((m, n)) if want_all else np.empty((1, n))
    Cv = C0.copy()
    for i in range(n):
        if Cv[i] < 0.0: Cv[i] = 0.0
        elif Cv[i] > 1.0: Cv[i] = 1.0
    trace = np.empty(m)
    if want_all: out[0] = Cv
    b = np.empty(n); a = np.empty(n); c = np.empty(n); rhs = np.empty(n)
    cp = np.empty(n); dp = np.empty(n)
    tc = tev[0]
    for k in range(m):
        if k > 0:
            tt = tev[k]
            ns = int(np.ceil((tt-tc)/h))
            if ns < 1: ns = 1
            hh = (tt-tc)/ns
            for _ in range(ns):
                x0 = _interp(tc, tgrid, cv)
                x1 = _interp(tc+hh, tgrid, cv)
                for i in range(n):
                    d0 = Dc*mid[i] - (kap*G[i] + lam*x0*w[i])
                    d1 = Dc*mid[i] - (kap*G[i] + lam*x1*w[i])
                    rhs[i] = Cv[i]*(1.0 + hh/2.0*d0) + hh*kap*G[i]
                    b[i] = 1.0 - hh/2.0*d1
                    a[i] = -hh/2.0*Dc*lo[i]
                    c[i] = -hh/2.0*Dc*up[i]
                for i in range(1, n):
                    rhs[i] += hh/2.0*Dc*lo[i]*Cv[i-1]
                for i in range(n-1):
                    rhs[i] += hh/2.0*Dc*up[i]*Cv[i+1]
                # Thomas
                cp[0] = c[0]/b[0]; dp[0] = rhs[0]/b[0]
                for i in range(1, n):
                    den = b[i] - a[i]*cp[i-1]
                    cp[i] = c[i]/den
                    dp[i] = (rhs[i] - a[i]*dp[i-1])/den
                Cv[n-1] = dp[n-1]
                for i in range(n-2, -1, -1):
                    Cv[i] = dp[i] - cp[i]*Cv[i+1]
                for i in range(n):
                    if Cv[i] < 0.0: Cv[i] = 0.0
                    elif Cv[i] > 1.0: Cv[i] = 1.0
                tc += hh
        if want_all: out[k] = Cv
    if not want_all: out[0] = Cv
    return out

@njit(cache=True, fastmath=False)
def integrate_inner(kap, lam, Dc, w, tev, C0, tgrid, cv, G, lo, up, mid, h, imask):
    """Same integration, but returns only the inner-region mean series and the final state."""
    n = C0.shape[0]; m = tev.shape[0]
    Cv = C0.copy()
    for i in range(n):
        if Cv[i] < 0.0: Cv[i] = 0.0
        elif Cv[i] > 1.0: Cv[i] = 1.0
    ser = np.empty(m)
    b = np.empty(n); a = np.empty(n); c = np.empty(n); rhs = np.empty(n)
    cp = np.empty(n); dp = np.empty(n)
    nin = 0
    for i in range(n):
        if imask[i]: nin += 1
    tc = tev[0]
    for k in range(m):
        if k > 0:
            tt = tev[k]
            ns = int(np.ceil((tt-tc)/h))
            if ns < 1: ns = 1
            hh = (tt-tc)/ns
            for _ in range(ns):
                x0 = _interp(tc, tgrid, cv)
                x1 = _interp(tc+hh, tgrid, cv)
                for i in range(n):
                    d0 = Dc*mid[i] - (kap*G[i] + lam*x0*w[i])
                    d1 = Dc*mid[i] - (kap*G[i] + lam*x1*w[i])
                    rhs[i] = Cv[i]*(1.0 + hh/2.0*d0) + hh*kap*G[i]
                    b[i] = 1.0 - hh/2.0*d1
                    a[i] = -hh/2.0*Dc*lo[i]
                    c[i] = -hh/2.0*Dc*up[i]
                for i in range(1, n):
                    rhs[i] += hh/2.0*Dc*lo[i]*Cv[i-1]
                for i in range(n-1):
                    rhs[i] += hh/2.0*Dc*up[i]*Cv[i+1]
                cp[0] = c[0]/b[0]; dp[0] = rhs[0]/b[0]
                for i in range(1, n):
                    den = b[i] - a[i]*cp[i-1]
                    cp[i] = c[i]/den
                    dp[i] = (rhs[i] - a[i]*dp[i-1])/den
                Cv[n-1] = dp[n-1]
                for i in range(n-2, -1, -1):
                    Cv[i] = dp[i] - cp[i]*Cv[i+1]
                for i in range(n):
                    if Cv[i] < 0.0: Cv[i] = 0.0
                    elif Cv[i] > 1.0: Cv[i] = 1.0
                tc += hh
        s = 0.0
        for i in range(n):
            if imask[i]: s += Cv[i]
        ser[k] = s/nin
    return ser, Cv
