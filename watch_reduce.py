#!/usr/bin/env python3
"""
watch_reduce.py -- reduce iharm2d_v4 ASCII dumps as they appear, then delete them.

Runs alongside harm.  Keeps disk bounded: only the newest few dumps and the
grid file survive.  Writes <out>_rom.npz / <out>_mad.npz incrementally, so the
files are always usable even if the run is interrupted.

    python3 watch_reduce.py --dumps ./dumps --out mad128 --rmax 60 &

Stops when <dumps>/../DONE exists and no unprocessed dumps remain.
"""
import argparse, glob, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reduce_iharm2d as R          # reuse the reader + reduction verbatim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", default="run")
    ap.add_argument("--rmax", type=float, default=60.0)
    ap.add_argument("--rho-th", type=float, default=None)
    ap.add_argument("--keep", type=int, default=2, help="newest N dumps left on disk")
    ap.add_argument("--poll", type=float, default=15.0)
    a = ap.parse_args()

    done_flag = os.path.join(os.path.dirname(os.path.abspath(a.dumps)), "DONE")
    state = dict(T=[], PHI=[], MDOT=[], SIG=[], C=[], RHO=[], MF=[], MX=[], TI=[])
    seen, ih, imax, rho_th, r1d, spin = set(), None, None, a.rho_th, None, None
    idx = lambda p: int(re.search(r"dump_(\d+)$", p).group(1))

    def flush():
        if not state["T"]:
            return
        o = np.argsort(np.array(state["T"]))
        t = np.array(state["T"])[o]
        np.savez_compressed(f"{a.out}_mad.npz",
                            t=t, phi=np.array(state["PHI"])[o],
                            mdot=np.array(state["MDOT"])[o],
                            sigma=np.array(state["SIG"])[o],
                            a=spin, r_isco=R.r_isco(spin), rho_th=rho_th)
        np.savez_compressed(f"{a.out}_rom.npz",
                            time=t, radius=r1d[ih:imax],
                            C=np.clip(np.array(state["C"])[o], 0, 1),
                            chi=np.array(state["PHI"])[o],
                            rho=np.array(state["RHO"])[o],
                            mass_flux=np.array(state["MF"])[o],
                            maxwell_stress=np.array(state["MX"])[o],
                            inflow_time=np.array(state["TI"])[o])

    print("watching", a.dumps, flush=True)
    idle = 0
    while True:
        files = sorted(R.find_dumps(a.dumps)) if glob.glob(os.path.join(a.dumps, "dump_[0-9]*")) else []
        pending = [p for p in files if p not in seen]
        # a dump is complete once a later one exists
        ready = pending[:-a.keep] if len(pending) > a.keep else []
        if not ready:
            if os.path.exists(done_flag):
                # final pass: everything is complete
                ready = pending
                if not ready:
                    break
            else:
                idle += 1
                time.sleep(a.poll)
                continue
        idle = 0
        for p in ready:
            try:
                d = R.read_dump(p)
            except Exception as e:                      # partially written dump
                print("skip", os.path.basename(p), e, flush=True)
                continue
            if ih is None:
                spin = d["a"]
                r1d = d["r"][:, 0] if d["r"].ndim > 1 else d["r"]
                ih = R.horizon_index(r1d, spin)
                imax = int(np.searchsorted(r1d, a.rmax))
                if rho_th is None:
                    n2 = d["rho"].shape[1]
                    rho_th = 0.05 * float(np.max(d["rho"][:, int(.45*n2):int(.55*n2)]))
                print(f"a={spin} ih={ih} imax={imax} rho_th={rho_th:.4g}", flush=True)
            q = R.reduce_dump(d, ih, rho_th)
            sl = slice(ih, imax)
            state["T"].append(d["t"]);   state["PHI"].append(q["phi"])
            state["MDOT"].append(q["mdot"]); state["SIG"].append(q["sigma"])
            state["C"].append(q["C"][sl]);   state["RHO"].append(q["rho_bar"][sl])
            state["MF"].append(q["mflux"][sl]); state["MX"].append(q["maxw"][sl])
            state["TI"].append(q["tin"][sl])
            seen.add(p)
            os.remove(p)
        flush()
        print(f"  reduced {len(state['T'])} dumps, t -> {state['T'][-1]:.0f}", flush=True)
        if os.path.exists(done_flag):
            break
    flush()
    print("watcher finished:", len(state["T"]), "dumps", flush=True)


if __name__ == "__main__":
    main()
