"""Independent finite-n checks of the numerical calculations and level identity."""
import csv
import json
import math
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from profiles import normalized_means, direct_mean_recurrence, mean_lower_bound

ROOT = Path(__file__).resolve().parent


def enumerate_small(n, H=6):
    """All parent-labelled PORT trees, with exact rational probabilities."""
    # A state is the parent tuple for vertices 1,...,N, with -1 at the root.
    trees = {(-1, 0): Fraction(1)}
    for N in range(2, n+1):
        nxt = defaultdict(Fraction)
        for parents, probability in trees.items():
            children = [0]*N
            for parent in parents[1:]:
                children[parent] += 1
            for parent in range(N):
                nxt[parents+(parent,)] += probability*Fraction(children[parent]+1, 2*N-1)
        trees = nxt
    expected = [Fraction(0) for _ in range(H+1)]
    expected_missing = [Fraction(0) for _ in range(H+1)]
    for parents, probability in trees.items():
        N = len(parents)
        children, depth, branch = [0]*N, [0]*N, [False]*N
        branch[1] = True
        for v in range(1,N):
            parent = parents[v]
            children[parent] += 1
            depth[v] = depth[parent]+1
            if v > 1:
                branch[v] = branch[parent]
        S, B, L, LP = ([0]*(H+2) for _ in range(4))
        for v in range(N):
            LP[depth[v]] += 1
            if branch[v]:
                B[depth[v]] += children[v]+1
            else:
                L[depth[v]] += 1
                S[depth[v]] += children[v]+1
        # Counts reconstructed independently from the weighted identity.
        assert S[0] == 2+L[1]
        assert LP[1]-L[1] == 1
        for h in range(1,H+1):
            assert S[h] == L[h]+L[h+1]
            assert S[h]+B[h] == LP[h]+LP[h+1]
        for h in range(H+1):
            expected[h] += probability*S[h]
            expected_missing[h] += probability*B[h]
    assert sum(trees.values()) == 1
    L = math.log(n)
    scales = np.array([2**h*math.factorial(h)/(math.sqrt(n)*L**h) for h in range(H+1)])
    return np.array(expected, dtype=float)*scales, np.array(expected_missing, dtype=float)*scales, len(trees)


def main():
    checks = []
    for n in [2,3,4,5,7,20,100,1000]:
        H = min(8,n-1)
        fast, direct = normalized_means(n,H), direct_mean_recurrence(n,H)
        error = float(np.max(np.abs(fast-direct)/np.maximum(np.abs(direct),1e-15)))
        assert error < 2e-10, (n,error)
        checks.append({"n":n,"H":H,"relative_error":error})
    enumerations = []
    for n in range(2,7):
        means,gaps,count = enumerate_small(n)
        assert np.allclose(means,normalized_means(n,6),rtol=2e-10,atol=1e-12)
        assert math.isclose(sum(gaps),mean_lower_bound(n,6,1),rel_tol=2e-10)
        enumerations.append({"n":n,"labelled_parent_trees":count})
    mc_zscores = []
    with (ROOT/'results/coupling_mc.csv').open() as f:
        for r in csv.DictReader(f):
            if r['p'] != '1':
                continue
            exact=mean_lower_bound(int(r['n']),int(r['H']),1)
            z=(float(r['mean'])-exact)/float(r['se'])
            mc_zscores.append({"n":int(r['n']),"H":int(r['H']),"standard_errors":z})
    # This reports sampling discrepancies; it is not a deterministic test gate.
    report={"mean_recurrence_checks":checks,"exact_enumerations":enumerations,
            "mc_l1_deviations":mc_zscores,
            "maximum_mc_deviation_in_standard_errors":max(abs(x['standard_errors']) for x in mc_zscores)}
    (ROOT/'results/validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
