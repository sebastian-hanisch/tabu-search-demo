"""Auswertung der Tabu-Search-Demo: ein Lauf gegen einen Hill-Climbing-Abstieg (beste Verbesserung, dieselbe Zugauswahl wie Tabu Search
ohne Gedächtnis) und Hill Climbing mit Neustarts (voller Rescan, wie in der Wurzel-Demo), gleiches Bewertungsbudget. Sweeps,
Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp). Ein Vorschlag ist ein
bewerteter Nachbar; Tabu Search bewertet die volle 2-opt-Nachbarschaft auf einen Schlag (rund n²/2 Bewertungen je Iteration) - viel
weniger, aber teurere Iterationen als die Kandidatenlisten-Familie (ILS, VNS)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import tabu_algorithm as TS
import tabu_constants as C
import tabu_scenario as S
import tabu_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    tenure: int = C.DEFAULT_TENURE
    budget: int = C.DEFAULT_BUDGET
    start: str = C.DEFAULT_START
    chain_seed: int = C.DEFAULT_CHAIN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


def make_start(settings, D):
    if settings.start == "nearest":
        return T.nearest_neighbor_tour(D)
    return T.random_tour(len(D), np.random.default_rng(settings.chain_seed))


def hill_climbing_restarts(D, budget, seed):
    """Hill Climbing mit Neustarts, voller Rescan (wie in der Wurzel-Demo)."""
    rng = np.random.default_rng(seed)
    used, starts, best = 0, 0, None
    while used < budget or best is None:
        cap = None if best is None else budget - used
        r = T.descend(D, T.random_tour(len(D), rng), "2opt", "first", keep_steps=False, max_evaluations=cap)
        used += r.evaluations
        starts += 1
        if best is None or r.length < best.length:
            best = r
    return best.tour, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    start_tour: np.ndarray
    run: object
    seconds: float
    hc: object                      # ein Hill-Climbing-Abstieg (beste Verbesserung) aus derselben Startlösung
    hc_seconds: float
    hcr_tour: np.ndarray             # Hill Climbing mit Neustarts, voller Rescan, gleiches Budget
    hcr_starts: int
    hcr_seconds: float
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(self.run.best_length)

    @property
    def final_gap(self):
        return self.gap_of(self.run.final_length)

    @property
    def hc_gap(self):
        return self.gap_of(self.hc.length)

    @property
    def hcr_gap(self):
        return self.gap_of(T.tour_length(self.hcr_tour, self.D))

    @property
    def start_gap(self):
        return self.gap_of(T.tour_length(self.start_tour, self.D))

    @property
    def override_rate(self):
        return self.run.overrides / max(self.run.iterations, 1)


def analyse(settings, keep_snapshots=True, with_hc=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    start = make_start(settings, D)
    t0 = time.perf_counter()
    run = TS.run(D, start, tenure=settings.tenure, budget=settings.budget, keep_snapshots=keep_snapshots)
    seconds = time.perf_counter() - t0
    hc = hc_tour = None
    hc_seconds = hcr_seconds = 0.0
    hcr_starts = 0
    if with_hc:
        t0 = time.perf_counter()
        hc = T.descend(D, start, "2opt", "best", keep_steps=False)
        hc_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        hc_tour, hcr_starts, _ = hill_climbing_restarts(D, settings.budget, settings.chain_seed)
        hcr_seconds = time.perf_counter() - t0
    return Analysis(settings, inst, D, bound, start, run, seconds, hc, hc_seconds, hc_tour, hcr_starts, hcr_seconds,
                     T.count_crossings(inst.xy, run.best_tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.5                  # so viel besser als Hill Climbing mit Neustarts gilt als Sieg (Prozentpunkte)
LOSE_MARGIN = 0.5                 # so viel schlechter gilt als Niederlage


def verdict(a):
    """Code: beats_hc (deutlich besser als Hill Climbing mit Neustarts, voller Rescan, bei gleichem Budget), hc_wins, comparable.
    Gilt für diesen einen Lauf - die Ketten streuen."""
    if a.gap <= a.hcr_gap - WIN_MARGIN:
        return "beats_hc"
    if a.hcr_gap <= a.gap - LOSE_MARGIN:
        return "hc_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch), keep_snapshots=False)
            rows.append({"gap": a.gap, "final": a.final_gap, "hc": a.hc_gap, "hcr": a.hcr_gap, "hcr_starts": a.hcr_starts, "seconds": a.seconds,
                         "override_rate": a.override_rate, "iterations": a.run.iterations, "crossings": a.crossings_end})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000), "tenure": (0, 3, 5, 10, 20, 50, 100, 200),
                "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100), "start": ("random", "nearest")}
SWEEP_LABELS = {"budget": "Budget (bewertete Nachbarn)", "tenure": "Tabu-Tenure", "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)",
                "start": "Startlösung"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


SCALING_N = C.SCALING_N
SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: Tabu Search (beste Tour) und ein Hill-Climbing-Abstieg aus derselben zufälligen Startlösung."""
    tabu_gaps, hc_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch, start="random"), keep_snapshots=False)
        tabu_gaps.append(a.gap)
        hc_gaps.append(a.hc_gap)
    return {"tabu": np.array(tabu_gaps), "hc": np.array(hc_gaps)}
