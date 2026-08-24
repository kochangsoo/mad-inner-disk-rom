#!/usr/bin/env python3
"""
reduce_iharm2d.py

iharm2d_v4 (ASCII 덤프) 런 하나를 1회 통과로 축약한다. 산출물 두 개:

  <out>_mad.npz   MAD 진단 시계열 (phi_BH, Mdot, sigma_rho)  -> 게이트 G1-G3
  <out>_rom.npz   ROM 게이트 입력 스키마                      -> 2d_gate_test_code

두 파일만 보관하면 원시 덤프는 삭제해도 된다.

사용법:
    python reduce_iharm2d.py --dumps ./dumps --inspect          # 1단계: 구조 확인
    python reduce_iharm2d.py --dumps ./dumps --out a0p9375      # 2단계: 축약

ROM 스키마 (2d_gate_test_code/README.md 요구사항):
    time (Nt,) / radius (Nr,) / C (Nt,Nr) / chi (Nt,)
    rho, mass_flux, maxwell_stress, inflow_time  각 (Nt,Nr)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------- I/O
#
# iharm2d_v4 writes ASCII dumps: "dumps/dump_%08d" (no extension, no HDF5),
# plus a single "dumps/grid" file.  The original HDF5 reader in reduce_iharm2d.py
# cannot open them.  Layout, read off core/io.c:
#
#   dump header (line 1, whitespace separated):
#       <problem data> VERSION has_electrons "grid" METRIC RECON
#       N1 N2 n_prims n_prims_passive gam cour tf
#       startx1 startx2 dx1 dx2 n_dim
#       [poly_xt poly_alpha mks_smooth]  Rin Rout Rhor Risco hslope a
#       t dt nstep dump_cnt DTd DTf
#   dump body: N1*N2 rows, ILOOP JLOOP order (i outer, j inner), 16 columns:
#       RHO UU U1 U2 U3 B1 B2 B3  jcon[0..3]  gamma  divB  fail  fflag
#
#   grid: N1*N2 rows, same order, 40 columns:
#       x y r th X1 X2 gdet lapse gcon[16] gcov[16]
#
_GRID_CACHE = {}


def find_dumps(d):
    f = sorted(glob.glob(os.path.join(d, "dump_[0-9]*")))
    if not f:
        f = sorted(glob.glob(os.path.join(d, "dump*.h5")))
    if not f:
        sys.exit(f"덤프를 찾지 못했습니다: {d}")
    return [p for p in f if not p.endswith("dump_abort")]


def _read_grid(dump_dir, N1, N2):
    key = os.path.abspath(dump_dir)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    gpath = os.path.join(dump_dir, "grid")
    if not os.path.exists(gpath):
        sys.exit(f"grid 파일이 없습니다: {gpath}")
    g = np.loadtxt(gpath)
    if g.shape[0] != N1 * N2:
        sys.exit(f"grid 행 수 {g.shape[0]} != N1*N2 {N1*N2}")
    out = dict(
        r=g[:, 2].reshape(N1, N2),
        th=g[:, 3].reshape(N1, N2),
        gdet=g[:, 6].reshape(N1, N2),
        lapse=g[:, 7].reshape(N1, N2),
        gcon=g[:, 8:24].reshape(N1, N2, 4, 4),
        gcov=g[:, 24:40].reshape(N1, N2, 4, 4),
    )
    _GRID_CACHE[key] = out
    return out


def _parse_header(line):
    tok = line.split()
    # anchor on the reconstruction string, which is followed by N1 N2 ...
    for name in ("WENO", "LINEAR"):
        if name in tok:
            k = tok.index(name)
            break
    else:
        sys.exit("헤더에서 RECONSTRUCTION 문자열을 찾지 못했습니다.")
    v = tok[k + 1:]
    N1, N2 = int(v[0]), int(v[1])
    n_prims = int(v[2])
    # v[3]=n_prims_passive, v[4]=gam, v[5]=cour, v[6]=tf,
    # v[7]=startx1 v[8]=startx2 v[9]=dx1 v[10]=dx2 v[11]=n_dim
    dx1, dx2 = float(v[9]), float(v[10])
    n_dim = int(float(v[11]))
    j = 12
    metric = "FMKS" if "FMKS" in tok else ("MKS" if "MKS" in tok else "MINKOWSKI")
    if metric == "FMKS":
        j += 3                                   # poly_xt, poly_alpha, mks_smooth
    if metric in ("MKS", "FMKS"):
        Rin, Rout, Rhor, Risco, hslope, a = (float(x) for x in v[j:j + 6])
        j += 6
    else:
        Rin = Rout = Rhor = Risco = hslope = a = np.nan
    t = float(v[j])
    return dict(N1=N1, N2=N2, n_prims=n_prims, dx1=dx1, dx2=dx2,
                a=a, Rhor=Rhor, Risco=Risco, Rout=Rout, t=t)


def inspect(path):
    h = _parse_header(open(path).readline())
    print(f"\n=== {os.path.basename(path)} ===")
    for k, val in h.items():
        print(f"  {k:12s} {val}")
    body = np.loadtxt(path, skiprows=1, max_rows=3)
    print(f"  body columns  {body.shape[1]}  (expect n_prims+4+2+2 = "
          f"{h['n_prims'] + 8})")


def read_dump(path):
    """iharm2d_v4 ASCII dump -> the dict that reduce_dump() expects."""
    with open(path) as fp:
        head = fp.readline()
    h = _parse_header(head)
    N1, N2, npr = h["N1"], h["N2"], h["n_prims"]
    body = np.loadtxt(path, skiprows=1)
    if body.shape[0] != N1 * N2:
        sys.exit(f"{path}: 본문 행 수 {body.shape[0]} != N1*N2 {N1*N2}")
    P = body[:, :npr].reshape(N1, N2, npr)       # ILOOP JLOOP -> (i, j)
    G = _read_grid(os.path.dirname(path) or ".", N1, N2)

    # HARM primitives U1..U3 are the relative 4-velocity \tilde{u}^i.
    # Recover u^r, which is what the mass flux needs:
    #     gamma = sqrt(1 + gcov_ij \tilde u^i \tilde u^j),  alpha = lapse
    #     u^i   = \tilde u^i - gamma * alpha * gcon^{0i}
    ut = P[..., 2:5]
    gcov_s = G["gcov"][..., 1:, 1:]
    qsq = np.einsum("ijmn,ijm,ijn->ij", gcov_s, ut, ut)
    gamma = np.sqrt(np.maximum(1.0 + qsq, 1.0))
    alpha = G["lapse"]
    ur = ut[..., 0] - gamma * alpha * G["gcon"][..., 0, 1]

    return dict(
        t=h["t"],
        rho=P[..., 0], u=P[..., 1],
        U1=ur, U2=P[..., 3], U3=P[..., 4],       # U1 replaced by u^r
        B1=P[..., 5], B2=P[..., 6], B3=P[..., 7],
        r=G["r"], gdet=G["gdet"], gcov=G["gcov"],
        dx1=h["dx1"], dx2=h["dx2"], a=h["a"],
    )


# ---------------------------------------------------------------- 유도량

def horizon_index(r1d, a):
    rp = 1.0 + np.sqrt(max(1.0 - a * a, 0.0))
    return int(np.clip(np.searchsorted(r1d, rp * 1.05), 1, len(r1d) - 2))


def r_isco(a):
    z1 = 1 + (1 - a**2) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = np.sqrt(3 * a**2 + z1**2)
    s = np.sign(a) if a != 0 else 1.0
    return 3 + z2 - s * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def theta_average(q, gdet, dx2):
    """theta 방향 gdet 가중 평균 -> (Nr,)"""
    w = gdet * dx2
    return np.sum(q * w, axis=1) / np.sum(w, axis=1)


def reduce_dump(d, ih, rho_th):
    """덤프 하나에서 반경 프로파일과 스칼라 진단량을 뽑는다."""
    rho, gdet, dx2 = d["rho"], d["gdet"], d["dx2"]

    # --- 지평선 스칼라 ---
    gh = gdet[ih, :]
    mdot = -np.sum(rho[ih, :] * d["U1"][ih, :] * gh) * dx2
    phi_raw = 0.5 * np.sum(np.abs(d["B1"][ih, :]) * gh) * dx2
    phi = phi_raw / np.sqrt(mdot) if mdot > 0 else np.nan

    # --- 반경 프로파일 ---
    rho_bar = theta_average(rho, gdet, dx2)

    # 연결성: 밀도 임계 초과 셀의 theta 평균 분율
    C = theta_average((rho > rho_th).astype(float), gdet, dx2)

    # 질량 플럭스 (반경별)
    mflux = -theta_average(rho * d["U1"], gdet, dx2)

    # Maxwell 응력 대용: -B^r B^phi (전체 계량 없이 좌표 성분 사용)
    maxw = -theta_average(d["B1"] * d["B3"], gdet, dx2)

    # 유입 시간 r / |v^r|
    vr = np.abs(theta_average(d["U1"], gdet, dx2))
    r1d = d["r"][:, 0] if d["r"].ndim > 1 else d["r"]
    with np.errstate(divide="ignore", invalid="ignore"):
        tin = np.where(vr > 0, r1d / vr, np.nan)

    # 난류 생존 지표: 적도 밴드 밀도 상대 표준편차
    n2 = rho.shape[1]
    band = rho[ih:min(ih + 120, rho.shape[0] - 1), int(0.4 * n2):int(0.6 * n2)]
    m = np.mean(band)
    sigma = float(np.std(band) / m) if m > 0 else np.nan

    return dict(mdot=mdot, phi=phi, sigma=sigma,
                rho_bar=rho_bar, C=C, mflux=mflux, maxw=maxw, tin=tin)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", required=True)
    ap.add_argument("--out", default="run")
    ap.add_argument("--spin", type=float, default=None)
    ap.add_argument("--rho-th", type=float, default=None,
                    help="연결성 밀도 임계. 미지정 시 첫 덤프 적도 최대밀도의 5%%")
    ap.add_argument("--rmax", type=float, default=50.0)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()

    files = find_dumps(args.dumps)
    print(f"덤프 {len(files)}개")

    if args.inspect:
        inspect(files[0])
        print("\n구조 확인 후 read_dump()를 맞추고 --inspect 없이 재실행하십시오.")
        return

    d0 = read_dump(files[0])
    a = d0["a"] if d0["a"] is not None else args.spin
    if a is None:
        sys.exit("스핀을 결정할 수 없습니다. --spin 지정 필요.")
    r1d = d0["r"][:, 0] if d0["r"].ndim > 1 else d0["r"]
    ih = horizon_index(r1d, a)
    imax = int(np.searchsorted(r1d, args.rmax))

    rho_th = args.rho_th
    if rho_th is None:
        n2 = d0["rho"].shape[1]
        rho_th = 0.05 * float(np.max(d0["rho"][:, int(0.45 * n2):int(0.55 * n2)]))
    print(f"a = {a}   r_+ index = {ih}   r_ISCO = {r_isco(a):.3f}   rho_th = {rho_th:.4g}")

    T, PHI, MDOT, SIG = [], [], [], []
    Cs, RHO, MF, MX, TI = [], [], [], [], []

    for k, p in enumerate(files):
        d = read_dump(p)
        q = reduce_dump(d, ih, rho_th)
        T.append(d["t"]); PHI.append(q["phi"])
        MDOT.append(q["mdot"]); SIG.append(q["sigma"])
        sl = slice(ih, imax)
        Cs.append(q["C"][sl]); RHO.append(q["rho_bar"][sl])
        MF.append(q["mflux"][sl]); MX.append(q["maxw"][sl]); TI.append(q["tin"][sl])
        if (k + 1) % 50 == 0:
            print(f"  {k+1}/{len(files)}")

    o = np.argsort(np.array(T))
    t = np.array(T)[o]
    phi = np.array(PHI)[o]
    mdot = np.array(MDOT)[o]
    sig = np.array(SIG)[o]
    C = np.array(Cs)[o]
    radius = r1d[ih:imax]

    # 연결성 [0,1] 보장
    C = np.clip(C, 0.0, 1.0)

    np.savez_compressed(f"{args.out}_mad.npz",
                        t=t, phi=phi, mdot=mdot, sigma=sig, a=a,
                        r_isco=r_isco(a), rho_th=rho_th)

    # ROM 스키마. chi = 정규화 지평선 자속
    np.savez_compressed(f"{args.out}_rom.npz",
                        time=t, radius=radius, C=C, chi=phi,
                        rho=np.array(RHO)[o],
                        mass_flux=np.array(MF)[o],
                        maxwell_stress=np.array(MX)[o],
                        inflow_time=np.array(TI)[o])

    meta = dict(spin=float(a), r_isco=float(r_isco(a)), rho_th=float(rho_th),
                n_dumps=len(files), t_range=[float(t[0]), float(t[-1])],
                radius_range=[float(radius[0]), float(radius[-1])],
                dump_dir=os.path.abspath(args.dumps))
    with open(f"{args.out}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n저장: {args.out}_mad.npz / {args.out}_rom.npz / {args.out}_meta.json")
    print("두 NPZ를 백업한 뒤에는 원시 덤프를 삭제해도 된다.")


if __name__ == "__main__":
    main()
