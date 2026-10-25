"""Simulations for the results chapter: sigma-scan, l-scan, and the 1989 model.

Results go to data/results/<name>.npz (+ .json with parameters). Every run stores the full
histogram of ln M and of the number of blocks per event, and the individual events above
`keep_above` (moment, blocks, time) for maximum-likelihood fits.

    python scripts/thesis/results_data.py            # all missing runs
    python scripts/thesis/results_data.py NAME ...   # selected runs
"""

import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bksoc.bk import BKChain, LAW_1989, LAW_1991

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "results"
MU_EDGES = np.arange(-20.0, 12.0001, 0.05)
MAXN = 1 << 16


def one_block_moment(l, alpha, sigma):
    w1 = 2 * l * l + 1
    return sigma / w1 * (1 + np.exp(np.pi * alpha / np.sqrt(w1 - alpha**2))) if alpha**2 < w1 else sigma


RUNS = {}
# 1) sigma-scan: alpha = 3, l = 6, N = 4096, nu -> 0
for sig in (0.1, 0.03, 0.01, 0.003, 0.001, 0.0003):
    RUNS[f"sig_{sig}"] = dict(law=LAW_1991, N=4096, l=6.0, alpha=3.0, sigma=sig, nu=0.0,
                              transient=200, measure=1000, keep_above=10 * one_block_moment(6, 3, sig))
# 2) l-scan towards the continuum limit, sigma = 0.01
RUNS["ell_12_N16384"] = dict(law=LAW_1991, N=16384, l=12.0, alpha=3.0, sigma=0.01, nu=0.0,
                             transient=200, measure=400, keep_above=10 * one_block_moment(12, 3, 0.01))
RUNS["ell_16"] = dict(law=LAW_1991, N=16384, l=16.0, alpha=3.0, sigma=0.01, nu=0.0,
                      transient=200, measure=300, keep_above=10 * one_block_moment(16, 3, 0.01))
RUNS["ell_24"] = dict(law=LAW_1991, N=32768, l=24.0, alpha=3.0, sigma=0.01, nu=0.0,
                      transient=200, measure=150, keep_above=10 * one_block_moment(24, 3, 0.01))
# 3) the 1989 model (symmetric friction, finite loading speed), parameters of Carlson & Langer Fig. 6.
#    Their run lengths T are in loading periods 2/nu, i.e. a plate displacement of 2 T.
FIG6 = {"a": dict(alpha=2.5, nu=0.01, N=100, T=1000), "b": dict(alpha=2.5, nu=0.001, N=100, T=320),
        "c": dict(alpha=2.5, nu=0.001, N=400, T=225), "d": dict(alpha=1.0, nu=0.001, N=100, T=45),
        "e": dict(alpha=4.0, nu=0.01, N=100, T=50)}
for key, p in FIG6.items():
    base = dict(law=LAW_1989, N=p["N"], l=10.0, alpha=p["alpha"], sigma=0.0, nu=p["nu"], transient=40,
                keep_above=0.0, fig=key)
    RUNS[f"cl89_{key}_long"] = dict(base, measure=max(4000, 4 * 2 * p["T"]), seed=0)
    for s in range(1, 6):
        RUNS[f"cl89_{key}_seed{s}"] = dict(base, measure=2 * p["T"], seed=s)


def run(name):
    p = RUNS[name]
    t0 = time.time()
    seed = p.get("seed", 0) * 1000 + zlib.crc32(name.encode()) % 1000
    m = BKChain(p["N"], p["l"], p["alpha"], p["sigma"], nu=p["nu"], law=p["law"], seed=seed)
    chunk = 20_000 if p["N"] <= 400 else 100_000
    while m.P < p["transient"]:
        m.run(chunk)
    P0, tau0 = m.P, m.tau
    h_mu = np.zeros(MU_EDGES.size - 1, dtype=np.int64)
    h_blocks = np.zeros(MAXN + 1, dtype=np.int64)
    keep = {k: [] for k in ("M", "blocks", "time", "duration")}
    n_events, total_M = 0, 0.0
    while m.P - P0 < p["measure"]:
        r = m.run(chunk)
        M = r["moment"]
        n_events += M.size
        total_M += M.sum()
        pos = M > 0
        h_mu += np.histogram(np.log(M[pos]), MU_EDGES)[0]
        h_blocks += np.bincount(np.minimum(r["blocks"], MAXN), minlength=MAXN + 1)
        sel = M > p["keep_above"]
        keep["M"].append(M[sel].astype(np.float64))
        keep["blocks"].append(r["blocks"][sel].astype(np.int32))
        keep["time"].append(r["time"][sel] - tau0)
        keep["duration"].append(r["duration"][sel].astype(np.float32))
    dP, dtau = m.P - P0, m.tau - tau0
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", mu_edges=MU_EDGES, h_mu=h_mu, h_blocks=h_blocks, dP=dP, dtau=dtau,
                        N=p["N"], **{k: np.concatenate(v) for k, v in keep.items()})
    meta = dict(p, name=name, events=int(n_events), plate_displacement=float(dP), time=float(dtau),
                sum_rule=float(total_M / (p["N"] * dP)), seconds=round(time.time() - t0, 1))
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=1))
    return meta


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in RUNS if not (OUT / f"{n}.npz").exists()]
    # longest runs first so that they do not end up running alone at the end
    order = ["ell_24", "ell_16", "ell_12_N16384"] + [n for n in names if n.startswith("sig")]
    names = [n for n in order if n in names] + [n for n in names if n not in order]
    with Pool(min(len(names), 10)) as pool:
        for meta in pool.imap_unordered(run, names):
            print(meta["name"], meta["events"], "events,", meta["seconds"], "s, sum rule",
                  round(meta["sum_rule"], 5), flush=True)
