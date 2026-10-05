"""Unabhängiges Orakel für Tabu Search: Schleifen-Neuimplementierung mit Listen und einem Wörterbuch der Entfernungszeitpunkte (statt der n×n-Sperrmatrix) auf ganzzahligen
Entfernungsmatrizen (exakte Gleichstände): Zug, Notfall-Überschreibung, Aspiration, Tourfolge, Bewertungszähler und Verlaufskurve müssen Schritt für Schritt übereinstimmen;
dazu Brute Force auf einem Mini-TSP (kein Ergebnis unter dem Optimum)."""

import itertools

import numpy as np
import pytest

import tabu_algorithm as TS
import tabu_tour as T


def _len(t, D):
    return sum(D[t[k]][t[(k + 1) % len(t)]] for k in range(len(t)))


def _reference(D, start, tenure, budget, trace_points=300):
    n = len(D)
    t, length = list(start), _len(start, D)
    best_len = length
    removed_at, ev, its, overrides = {}, 0, 0, 0
    moves, snaps = [], [list(t)]
    every = max(1, budget // trace_points)
    next_trace, trace = every, [(0, length, length)]
    while ev < budget:
        k = its + 1
        valid, allowed = [], []
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b, c, d = t[i], t[i + 1], t[j], t[(j + 1) % n]
                delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
                created = (frozenset((a, c)), frozenset((b, d)))
                tabu = any(e in removed_at and k - removed_at[e] <= tenure for e in created)   # eine entfernte Kante bleibt für die `tenure` folgenden Iterationen gesperrt
                valid.append((delta, i, j))
                if not tabu or length + delta < best_len - 1e-9:
                    allowed.append((delta, i, j))
        pool = allowed or valid
        m = min(p[0] for p in pool)
        delta, i, j = next(p for p in pool if p[0] == m)
        ev += len(valid)
        its += 1
        overrides += not allowed
        moves.append((i, j, not allowed))
        removed_at[frozenset((t[i], t[i + 1]))] = its
        removed_at[frozenset((t[j], t[(j + 1) % n]))] = its
        t = t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]
        length += delta
        best_len = min(best_len, length)
        snaps.append(list(t))
        if ev >= next_trace or ev >= budget:
            trace.append((ev, length, best_len))
            next_trace = (ev // every + 1) * every
    return ev, its, overrides, moves, snaps, best_len, trace


def test_run_matches_a_loop_reimplementation_step_by_step():
    rng = np.random.default_rng(5)
    for it in range(40):
        n = int(rng.integers(5, 12))
        M = np.triu(rng.integers(0, int(rng.choice([3, 30])) + 1, size=(n, n)), 1)
        D = (M + M.T).astype(float)
        start = [0] + rng.permutation(np.arange(1, n)).tolist()
        tenure = int(rng.choice([0, 1, 3, 8, 1000]))
        per = n * (n - 3) // 2
        budget = int(rng.integers(1, 40)) * per + int(rng.integers(0, per))
        got = TS.run(D, np.array(start), tenure=tenure, budget=budget, keep_snapshots=True, debug_trace=True)
        ev, its, overrides, moves, snaps, best_len, trace = _reference(D, start, tenure, budget)
        assert (got.evaluations, got.iterations, got.overrides) == (ev, its, overrides) and ev == its * per
        assert [tuple(m) for m in got.debug] == moves
        assert len(got.snapshots) == len(snaps) and all(list(a) == b for a, b in zip(got.snapshots, snaps))
        assert got.best_length == pytest.approx(best_len, abs=1e-9) == pytest.approx(min(_len(s, D) for s in snaps), abs=1e-9)
        assert np.array_equal(got.trace_iter, [x[0] for x in trace]) and np.allclose(got.trace_length, [x[1] for x in trace]) and np.allclose(got.trace_best, [x[2] for x in trace])
        assert got.evaluations - per < budget <= got.evaluations           # eine Iteration weniger hätte das Budget nicht erreicht


def test_the_trace_has_many_points_because_it_is_counted_in_evaluations_not_iterations():
    rng = np.random.default_rng(1)
    D = T.dist_matrix(rng.random((30, 2)) * 100)
    r = TS.run(D, T.random_tour(30, rng), tenure=10, budget=60000, keep_snapshots=False)
    assert r.iterations > 100 and len(r.trace_iter) > 100 and np.all(np.diff(r.trace_iter) > 0) and r.trace_iter[-1] == r.evaluations


def test_never_below_the_brute_force_optimum_on_a_mini_tsp_and_usually_reaches_it():
    rng = np.random.default_rng(2)
    hits = 0
    for _ in range(15):
        D = T.dist_matrix(rng.random((7, 2)) * 100)
        opt = min(T.tour_length([0, *p], D) for p in itertools.permutations(range(1, 7)))
        r = TS.run(D, T.random_tour(7, rng), tenure=3, budget=200 * 14, keep_snapshots=False)
        assert r.best_length >= opt - 1e-9
        hits += r.best_length <= opt + 1e-9
    assert hits >= 12
