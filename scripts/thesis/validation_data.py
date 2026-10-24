"""Statistical convergence of the BK solver: the distribution D(mu) for two step sizes.

    python scripts/thesis/validation_data.py
"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from bksoc.bk import BKChain

OUT = Path(__file__).resolve().parents[2] / "data" / "validation"


def job(c):
    l = 6.0
    h = c / np.sqrt(4 * l * l + 1)
    m = BKChain(1024, l, 3.0, 0.01, h=h, seed=5)
    while m.P < 100:
        m.run(100_000)
    P0, Ms = m.P, []
    while m.P - P0 < 1000:
        Ms.append(m.run(200_000)["moment"])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"h_{c}.npz", M=np.concatenate(Ms), dP=m.P - P0, N=1024, c=c)
    return c


def seed_job(args):
    """Rates of one-block, all, and large events per unit plate displacement, for one seed and step."""
    c, seed = args
    l = 6.0
    m = BKChain(1024, l, 3.0, 0.01, h=c / np.sqrt(4 * l * l + 1), seed=seed)
    while m.P < 100:
        m.run(100_000)
    P0, n1, ntot, big = m.P, 0, 0, 0
    while m.P - P0 < 600:
        M = m.run(100_000)["moment"]
        n1 += int(np.sum(M < 6e-4))
        ntot += M.size
        big += int(np.sum(M > 6.0))
    dP = m.P - P0
    return dict(c=c, seed=seed, one_block=n1 / dP, all=ntot / dP, large=big / dP)


if __name__ == "__main__":
    import json
    with Pool(6) as p:
        print(p.map(job, [0.1, 0.05]))
        res = p.map(seed_job, [(c, s) for c in (0.1, 0.05) for s in (1, 2, 3)])
    (OUT / "seeds.json").write_text(json.dumps(res, indent=1))
