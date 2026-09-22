"""tabu_algorithm.run: Budget-Buchführung (volle Nachbarschaft je Iteration), Tabu-Kanten-Verwaltung (Ablauf nach genau `tenure`
Iterationen, Aspiration, Notfall-Überschreibung), Regressionsschutz, Determinismus, "ohne Tabu"-Ablation (tenure=0 pendelt sofort)."""

import numpy as np
import pytest

import tabu_algorithm as TS
import tabu_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


def test_tour_stays_valid_and_length_matches_recomputed_length():
    D = _instance(20, 1)
    start = T.random_tour(20, np.random.default_rng(0))
    r = TS.run(D, start, tenure=10, budget=20000)
    assert sorted(r.best_tour.tolist()) == list(range(20))
    assert sorted(r.final_tour.tolist()) == list(range(20))
    assert r.best_length == pytest.approx(T.tour_length(r.best_tour, D), abs=1e-6)
    assert r.final_length == pytest.approx(T.tour_length(r.final_tour, D), abs=1e-6)


def test_budget_is_respected_and_exceeded_by_at_most_one_full_neighborhood():
    D = _instance(25, 2)
    start = T.random_tour(25, np.random.default_rng(1))
    budget = 20000
    r = TS.run(D, start, tenure=15, budget=budget)
    n = len(D)
    max_neighborhood = n * (n - 3) // 2 + n                  # grosszuegige obere Schranke fuer eine volle 2-opt-Nachbarschaft
    assert r.evaluations >= budget
    assert r.evaluations <= budget + max_neighborhood
    assert r.iterations >= 1


def test_best_length_is_never_worse_than_the_start():
    D = _instance(30, 3)
    start = T.random_tour(30, np.random.default_rng(2))
    start_length = T.tour_length(start, D)
    r = TS.run(D, start, tenure=20, budget=30000)
    assert r.best_length <= start_length + 1e-6


def test_the_search_takes_the_best_allowed_move_an_independent_replay_of_the_tabu_rule_confirms_it():
    """Unabhaengige Nachrechnung: aus den Momentaufnahmen + dem debug_trace (i, j, override) wird die Tabu-Liste NUR
    anhand der oeffentlichen Regel ("die zwei entfernten Kanten sind fuer `tenure` Iterationen tabu") neu aufgebaut -
    unabhaengig von der internen `tabu_until`-Datenstruktur des Algorithmus. Bei jeder Iteration ohne Notfall-
    Ueberschreibung muss der gewaehlte Zug der beste unter den so rekonstruiert ERLAUBTEN Zuegen sein."""
    D = _instance(16, 4)
    start = T.random_tour(16, np.random.default_rng(3))
    tenure = 5
    r = TS.run(D, start, tenure=tenure, budget=8000, keep_snapshots=True, debug_trace=True)
    n = len(D)
    tabu_until = {}                                             # (min(u,v), max(u,v)) -> Iteration, bis zu der die Kante tabu ist
    best_so_far = T.tour_length(r.snapshots[0], D)
    checked = 0
    for it, (i, j, override) in enumerate(r.debug[:200]):        # die ersten 200 Iterationen genuegen fuer eine gruendliche Pruefung
        cur = r.snapshots[it]
        delta, valid = T._delta_2opt(cur, D)
        cur_len = T.tour_length(cur, D)
        nxt = np.roll(cur, -1)
        is_tabu = np.zeros_like(valid)
        for (u, v), expiry in tabu_until.items():
            if expiry <= it:
                continue
            is_tabu |= (cur[:, None] == u) & (cur[None, :] == v)
            is_tabu |= (cur[:, None] == v) & (cur[None, :] == u)
            is_tabu |= (nxt[:, None] == u) & (nxt[None, :] == v)
            is_tabu |= (nxt[:, None] == v) & (nxt[None, :] == u)
        aspiring = (cur_len + delta) < best_so_far - 1e-9
        allowed = valid & (~is_tabu | aspiring)
        if not override:
            assert allowed[i, j]
            best_allowed = float(np.min(np.where(allowed, delta, np.inf)))
            assert delta[i, j] <= best_allowed + 1e-6
            checked += 1
        else:
            assert not allowed.any()
        removed_a = (int(cur[i]), int(cur[i + 1]))
        removed_b = (int(cur[j]), int(cur[(j + 1) % n]))
        tabu_until[(min(removed_a), max(removed_a))] = it + 1 + tenure
        tabu_until[(min(removed_b), max(removed_b))] = it + 1 + tenure
        new_len = cur_len + float(delta[i, j])
        if new_len < best_so_far - 1e-9:
            best_so_far = new_len
    assert checked > 10                                         # der Test hat tatsaechlich etwas geprueft, nicht nur Overrides gesehen


def test_no_tabu_eventually_oscillates_between_two_tours_once_stuck():
    """tenure=0: keine Kante bleibt gesperrt. Anfangs steigt die Suche normal ab (viele verbessernde Zuege, viele
    verschiedene Touren); ist sie einmal in einem lokalen Optimum, macht der naechste (verschlechternde) Zug den eben
    gemachten sofort wieder rueckgaengig (derselbe Zug ist wieder der beste), und die Suche pendelt fortan zwischen
    genau zwei Touren, ohne weiteren Fortschritt."""
    D = _instance(14, 6)
    start = T.random_tour(14, np.random.default_rng(5))
    r = TS.run(D, start, tenure=0, budget=4000, keep_snapshots=True)
    tail = [tuple(np.asarray(s).tolist()) for s in r.snapshots[-20:]]
    assert len(set(tail)) <= 2                                  # am Ende des Laufs: nur noch zwei pendelnde Touren
    assert tail[-1] == tail[-3] and tail[-2] == tail[-4]         # echtes Pendeln, keine zufaellige Wiederholung


def test_deterministic_given_the_same_start_and_different_for_another_start():
    """Tabu Search selbst ist deterministisch (kein Zufall im Kern) - derselbe Lauf mit derselben Startloesung muss
    IMMER dasselbe Ergebnis liefern; nur die Startloesung (die Kette in der Auswertungsschicht) entscheidet."""
    D = _instance(25, 7)
    start_a = T.random_tour(25, np.random.default_rng(6))
    start_c = T.random_tour(25, np.random.default_rng(20))
    a = TS.run(D, start_a, tenure=10, budget=10000)
    b = TS.run(D, start_a, tenure=10, budget=10000)
    c = TS.run(D, start_c, tenure=10, budget=10000)
    assert np.array_equal(a.best_tour, b.best_tour) and a.evaluations == b.evaluations
    assert not np.array_equal(a.best_tour, c.best_tour) or a.evaluations != c.evaluations


def test_very_high_tenure_triggers_overrides():
    """Bei einer Tenure, die groesser als die Instanz ist, muessen irgendwann ALLE gueltigen Zuege tabu sein -
    die Notfall-Ueberschreibung muss dann greifen, sonst bliebe die Suche stecken."""
    D = _instance(12, 8)
    start = T.random_tour(12, np.random.default_rng(7))
    r = TS.run(D, start, tenure=10_000, budget=5000)
    assert r.overrides > 0


def test_zero_tenure_never_overrides():
    D = _instance(20, 9)
    start = T.random_tour(20, np.random.default_rng(8))
    r = TS.run(D, start, tenure=0, budget=5000)
    assert r.overrides == 0


def test_negative_tenure_is_rejected():
    D = _instance(10, 0)
    start = T.random_tour(10, np.random.default_rng(0))
    with pytest.raises(ValueError):
        TS.run(D, start, tenure=-1, budget=1000)


def test_snapshots_are_only_kept_when_requested():
    D = _instance(12, 0)
    start = T.random_tour(12, np.random.default_rng(0))
    with_snaps = TS.run(D, start, tenure=10, budget=3000, keep_snapshots=True)
    without = TS.run(D, start, tenure=10, budget=3000, keep_snapshots=False)
    assert len(with_snaps.snapshots) == with_snaps.iterations + 1
    assert without.snapshots == []
