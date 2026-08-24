# mad-inner-disk-rom

Data and code for **"Testing a reduced model of transient inner-disk dynamics in
magnetically arrested black-hole accretion"** (C. Ko, submitted).

A minimal reaction–diffusion closure for a bounded connectivity order parameter is
driven by the horizon magnetic flux measured in a GRMHD simulation, and asked whether
it predicts the inner-region response better than its own memoryless limit. This
repository contains the reduced GRMHD diagnostics, the null tests, the predictive
test with its surrogate null, and the scripts that produce every figure and number in
the paper.

## What is here

| File | Purpose |
|---|---|
| `mad_rom.npz` | reduced GRMHD output: `time`, `radius`, `C(r,t)`, `chi`, `rho`, `mass_flux`, `maxwell_stress`, `inflow_time` |
| `mad_mad.npz` | MAD diagnostics: `t`, `phi` (horizon flux), `mdot`, `sigma`, `a`, `r_isco`, `rho_th` |
| `param.dat` | `iharm2d_v4` parameter file for the simulation |
| `reduce_iharm2d.py`, `watch_reduce.py` | reduction pipeline: ASCII dumps → the two `.npz` files |
| `core.py` | data loader, reference (SciPy) integrator, fitting routines |
| `fastcore.py` | numba integrator, agreeing with the reference to 3.3e-16 in `C`, ~60× faster |
| `refit_null.py` | the predictive test of Sect. 6, with a full refit under every surrogate |
| `nonstat.py` | the same protocol on the non-stationary interval (negative control) |
| `selftest.py` | recovers known coefficients from synthetic data |
| `driven_test.py` | earlier standalone implementation, kept for cross-checking |
| `diag.py`, `fig4.py`, `fig_crt.py`, `fig_stat.py`, `fig_local.py` | figures 1–5 |
| `matplotlibrc` | Arial/Helvetica-metric fonts, TrueType embedding |
| `results/*.json` | outputs of every run quoted in the paper |

`mad_rom.npz` and `mad_mad.npz` are the files named `mad128_rom.npz` and
`mad128_mad.npz` elsewhere; they are renamed here to the names the scripts expect.

## Reproducing the paper

```bash
pip install -r requirements.txt
python3 selftest.py                                    # protocol recovers known coefficients
python3 refit_null.py 7900 5.0 0.5 500 2026 out.json   # primary result: S_loc, p
python3 nonstat.py 1200 40                             # negative control
for f in diag fig4 fig_crt fig_stat fig_local; do python3 $f.py; done
```

`refit_null.py` takes `T0 r_cut train_fraction N_surrogates seed out.json [scalar|localised|both]`.
The defaults above give the primary result of Sect. 6.2. Every entry of the
aggregation-radius and train/test-split tables is one further invocation with `r_cut`
or `train_fraction` changed; `results/` holds the outputs of all of them.

For each driver realisation — the real one and each of the *N* surrogates — the script
refits `(kappa, lambda, D_c[, ell_chi])` on the training window, refits the two-constant
memoryless limit on the same window with the same driver, integrates across the held-out
window, and evaluates the skill. The *p*-value is the rank of the real skill in that
refitted null, `(count + 1) / (N + 1)`.

**Runtime.** The primary 500-surrogate run takes roughly 20 minutes per model on one
core. `numba` compiles the integrator on first call (~1 s).

**Raw dumps are not archived.** They are reduced and deleted as the simulation runs.
The reduction is deterministic, and the run is reproducible from `param.dat` and
`iharm2d_v4`.

## Requirements

Python 3.10+, `numpy`, `scipy`, `matplotlib`, `numba`. The simulation itself was run
with the public code [`iharm2d_v4`](https://github.com/AFD-Illinois/iharm2d_v4)
(Prather et al. 2021), which is not redistributed here.

## Licence

Code is released under the MIT Licence (`LICENSE`). The reduced data files
(`mad_rom.npz`, `mad_mad.npz`, `results/*.json`) are released under
CC BY 4.0 (`LICENSE-DATA`).

## Citation

See `CITATION.cff`. Please cite the paper, and the archived release DOI if you use the
data or code.
