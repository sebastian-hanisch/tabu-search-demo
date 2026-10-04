"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen (je drei Ketten-Seeds) belegt.
Positive UND negative Aussagen: Tabu Search braucht bei knappem Budget deutlich mehr Vorschläge als die Geschwister-Demos, um überhaupt
konkurrenzfähig zu werden - das steht hier ebenso als Test wie die Stellen, an denen es gewinnt. Rechenzeiten sind nur als Größenordnung geprüft."""

from functools import lru_cache

import numpy as np
import pytest

import tabu_constants as C
import tabu_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Standardfall -------------------------------------------------------------------------------------------------------------------------------


def test_standard_case_numbers():
    std = cfg()
    near(std["gap"], 4.19, 1.5)
    near(std["hc"], 7.85, 1.0)
    near(std["hcr"], 4.88, 1.0)


# --- Budget-Sweep: die "teure Iteration"-Geschichte ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("budget,gap,tol", [
    (10000, 287.8, 100.0), (25000, 155.1, 60.0), (50000, 54.7, 25.0), (100000, 7.9, 3.0),
    (200000, 4.19, 1.5), (500000, 3.09, 1.2), (1000000, 2.31, 1.0), (2000000, 1.94, 1.0),
])
def test_budget_sweep_numbers(budget, gap, tol):
    near(cfg(budget=budget)["gap"], gap, tol)


def test_small_budgets_are_catastrophically_worse_than_a_single_descent():
    for budget in (10000, 25000, 50000):
        row = cfg(budget=budget)
        assert row["gap"] > row["hc"] + 20.0                       # weit schlechter als ein einzelner Hill-Climbing-Abstieg


def test_tabu_search_matches_a_single_descent_around_one_hundred_thousand_and_beats_restarts_from_two_hundred_thousand():
    at_100k = cfg(budget=100000)
    near(at_100k["gap"], at_100k["hc"], 2.0)                        # ungefaehr gleichauf mit einem einzelnen Abstieg
    at_200k = cfg(budget=200000)
    assert at_200k["gap"] < at_200k["hcr"] - 0.3                    # klar besser als Neustarts (voller Rescan)


def test_the_advantage_over_restarts_narrows_again_at_very_large_budgets():
    mid = cfg(budget=200000)
    large = cfg(budget=2000000)
    mid_edge = mid["hcr"] - mid["gap"]
    large_edge = large["hcr"] - large["gap"]
    assert mid_edge > 0 and large_edge < mid_edge + 0.5             # der Vorsprung ist bei 2M nicht groesser als bei 200T (naehert sich an oder dreht sich)


# --- Tenure-Sweep -----------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tenure,gap,tol", [(0, 7.54, 2.5), (5, 6.12, 2.0), (10, 5.18, 2.0), (20, 4.19, 1.5), (50, 4.52, 1.8), (200, 4.52, 1.8)])
def test_tenure_sweep_numbers_at_two_hundred_thousand(tenure, gap, tol):
    near(cfg(tenure=tenure)["gap"], gap, tol)


def test_tenure_twenty_beats_both_no_memory_and_very_long_tenure():
    zero = cfg(tenure=0)["gap"]
    twenty = cfg(tenure=20)["gap"]
    long_ = cfg(tenure=200)["gap"]
    assert twenty < zero and twenty < long_


def test_no_tabu_matches_a_single_hill_climbing_descent():
    near(cfg(tenure=0)["gap"], cfg(tenure=0)["hc"], 1.0)


def test_zero_tenure_never_triggers_the_emergency_override():
    assert cfg(tenure=0)["override_rate"] == 0.0


# --- Große Instanz ----------------------------------------------------------------------------------------------------------------------------


def test_large_instance_scales_far_worse_than_the_candidate_list_family():
    row = ev.run_config(ev.Settings(n=200), budget=1000000)
    near(row["gap"], 307.0, 100.0)
    assert row["gap"] > row["hcr"] + 100.0


# --- Startlösung ------------------------------------------------------------------------------------------------------------------------------


def test_a_good_start_solution_clearly_helps_unlike_ils_and_vns():
    random_ = cfg(start="random")
    nearest = cfg(start="nearest")
    near(nearest["gap"], 1.50, 1.0)
    near(random_["gap"], 4.19, 1.5)
    assert nearest["gap"] < random_["gap"] - 1.0                   # anders als bei ILS/VNS zaehlt die Startloesung hier deutlich


def test_bound_matches_the_frozen_reference():
    near(ev.reference_bound(60, 0, 100000), 618.76, 0.1)
