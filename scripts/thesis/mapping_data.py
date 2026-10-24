"""Simulations comparing the Burridge-Knopoff chain with its sandpile limit.

Every run discards a transient (in units of plate displacement) and then accumulates
histograms of ln M (moment), of the number of blocks per event and of the number of
topplings. Results go to data/mapping/<name>.npz with the parameters in <name>.json.

    python scripts/thesis/mapping_data.py            # all runs, in parallel
    python scripts/thesis/mapping_data.py bk_a3_l6   # selected runs
"""

import json
import zlib
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bksoc.bk import BKChain
from bksoc.mapped import MappedChain, RULE_FIXED, RULE_RESET, bk_one_block_drop

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "mapping"

MU_EDGES = np.arange(-16.0, 12.0001, 0.05)
MAXN = 1 << 15

RUNS = {}
for l in (6, 8, 12):
    N = 4096 if l < 10 else 8192
    RUNS[f"bk_a3_l{l}"] = dict(model="bk", N=N, l=l, alpha=3.0, sigma=0.01, transient=200, measure=1000)
    d = bk_one_block_drop(l, 3.0, 0.01)
    RUNS[f"fixed_a3_l{l}"] = dict(model="fixed", N=N, l=l, alpha=3.0, sigma=0.01, d=d, transient=100, measure=300)
    RUNS[f"reset_a3_l{l}"] = dict(model="reset", N=N, l=l, alpha=3.0, sigma=0.01, d=d, transient=100, measure=300)
for l in (6, 10, 14):
    N = 4096 if l < 10 else 8192
    RUNS[f"bk_a1_l{l}"] = dict(model="bk", N=N, l=l, alpha=1.0, sigma=0.01, transient=200, measure=500)
    d = bk_one_block_drop(l, 1.0, 0.01)
    RUNS[f"fixed_a1_l{l}"] = dict(model="fixed", N=N, l=l, alpha=1.0, sigma=0.01, d=d, transient=100, measure=300)
# threshold noise (heterogeneous friction) in the sandpile limit
for l in (6, 12):
    N = 4096 if l < 10 else 8192
    d = bk_one_block_drop(l, 3.0, 0.01)
    for w in (0.01, 0.05, 0.2):
        for model in ("fixed", "reset"):
            RUNS[f"noise_{model}_w{w}_l{l}"] = dict(model=model, N=N, l=l, alpha=3.0, sigma=0.01, d=d, noise=w,
                                                   transient=100, measure=300)


def make(p, seed):
    if p["model"] == "bk":
        return BKChain(p["N"], p["l"], p["alpha"], p["sigma"], seed=seed)
    rule = RULE_FIXED if p["model"] == "fixed" else RULE_RESET
    return MappedChain(p["N"], p["l"], rule, d=p["d"], f_res=1 - p["d"], seed=seed, noise=p.get("noise", 0.0))


def run(name):
    p = RUNS[name]
    t0 = time.time()
    m = make(p, seed=zlib.crc32(name.encode()))       # deterministic, reproducible seed
    chunk = 200_000
    while m.P < p["transient"]:
        m.run(chunk)
    P0 = m.P
    h_mu = np.zeros(MU_EDGES.size - 1, dtype=np.int64)
    h_blocks = np.zeros(MAXN + 1, dtype=np.int64)
    h_top = np.zeros(1, dtype=np.int64)
    n_events, total_M, low_mu = 0, 0.0, 0
    keep_M, keep_B = [], []
    while m.P - P0 < p["measure"]:
        r = m.run(chunk)
        M = r["moment"]
        blocks = r["blocks"] if p["model"] == "bk" else r["area"]
        n_events += M.size
        total_M += M.sum()
        mu = np.log(M)
        low_mu += np.count_nonzero(mu < MU_EDGES[0])
        h_mu += np.histogram(mu, MU_EDGES)[0]
        h_blocks += np.bincount(np.minimum(blocks, MAXN), minlength=MAXN + 1)
        if p["model"] != "bk":
            bc = np.bincount(r["size"])
            if bc.size > h_top.size:
                h_top = np.concatenate([h_top, np.zeros(bc.size - h_top.size, dtype=np.int64)])
            h_top[: bc.size] += bc
        else:
            keep_M.append(M.astype(np.float32))
            keep_B.append(blocks.astype(np.int32))
    dP = m.P - P0
    extra = {}
    if keep_M:
        extra = dict(M=np.concatenate(keep_M), blocks=np.concatenate(keep_B))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", mu_edges=MU_EDGES, h_mu=h_mu, h_blocks=h_blocks,
                        h_topplings=h_top, dP=dP, N=p["N"], **extra)
    meta = dict(p, name=name, events=int(n_events), plate_displacement=float(dP),
                sum_rule=float(total_M / (p["N"] * dP)), below_hist=int(low_mu),
                seconds=round(time.time() - t0, 1))
    (OUT / f"{name}.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in RUNS if not (OUT / f"{n}.npz").exists()]
    with Pool(min(len(names), 12)) as pool:
        for meta in pool.imap_unordered(run, names):
            print(meta["name"], meta["events"], "events,", meta["seconds"], "s, sum rule", round(meta["sum_rule"], 6),
                  flush=True)
