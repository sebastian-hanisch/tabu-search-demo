"""Konstanten der Tabu-Search-Demo: Szenario (wortgleich zur hill-climbing-demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
DEFAULT_START = "random"
START_LABELS = {"random": "Zufällig", "nearest": "Nächster Nachbar"}
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- Tabu-Search-eigene Regler --------------------------------------------------------------------------------
TENURE_MIN, TENURE_MAX, DEFAULT_TENURE = 0, 200, 20
DEFAULT_BUDGET = 200000

# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), 60 gleichverteilte Stopps, Abstand zur Schranke (2026-09-22):
# Tenure-Sweep (Budget 200T / 1M): 0 -> 7.54/7.54 %, 3 -> 6.45/6.45 %, 5 -> 6.12/6.12 %, 10 -> 5.18/4.51 %, 20 -> 4.19/2.31 % (BESTES gemessen bei beiden Budgets),
#   50 -> 4.52/3.41 %, 100 -> 4.52/4.52 %, 200 -> 4.52/4.52 % (Plateau: bei nur ~110-570 Iterationen laeuft eine so lange Sperre praktisch nie ab).
#   Ein klarer Sweet Spot bei 20, nicht "laenger ist sicherer" - dieselbe Lehre wie SA's Temperatur, ILS' Stoerstaerke, VNS' k_max.
# Budget-Sweep (Tenure=20, VOLLER Sweep 10T..2M): 10T 287.8 %, 25T 155.1 %, 50T 54.7 %, 100T 7.9 % (~= ein Abstieg, 7.85 %), 200T 4.2 %, 500T 3.1 %, 1M 2.3 %, 2M 1.9 %.
#   Jede Iteration bewertet die VOLLE Nachbarschaft (~n²/2 ~ 1800 Bewertungen bei 60 Stopps) statt einer billigen Kandidatenliste (ILS/VNS) oder eines
#   einzelnen Vorschlags (SA) - bei 10-50 Tausend Vorschlägen reichen nur 6-29 Iterationen, zu wenig, um von einer zufälligen Startlösung auch nur in die
#   Nähe eines Optimums zu kommen (katastrophal: 55-288 % über der Schranke). ERST ab ~100 Tausend erreicht Tabu Search die Güte eines einzelnen Hill-
#   Climbing-Abstiegs, ERST ab 200 Tausend schlägt es Hill Climbing mit Neustarts (voller Rescan) klar; bei 2 Millionen liegt es mit den Neustarts gleichauf.
#   Das ist die deutlichste "teure Iterationen"-Geschichte der ganzen Linie - kein anderes Stück braucht ein derart hohes Mindestbudget, um überhaupt
#   konkurrenzfähig zu werden. Bei 200 Stopps (n²/2 ~ 20 Tausend je Iteration) ist die Lücke bei 1 Million Vorschlägen (nur ~50 Iterationen) noch 307 %.
# "Ohne Tabu" (tenure=0) landet fast exakt bei einem einzelnen Hill-Climbing-Abstieg (6.72 % gegen 6.72 %) - ohne Gedächtnis kommt die Suche über das
#   erste lokale Optimum kaum hinaus (pendelt danach zwischen zwei Touren, siehe Tests), bringt aber keinen Schaden gegenüber einem reinen Abstieg.


def _preset(tenure=DEFAULT_TENURE, budget=DEFAULT_BUDGET, n=DEFAULT_N, start=DEFAULT_START):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "tenure": tenure, "budget": budget, "start": start, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Zu kleines Budget (25 Tausend)": _preset(budget=25000),
    "Ohne Tabu (Tenure 0)": _preset(tenure=0),
    "Zu lange Tenure (200)": _preset(tenure=200),
    "Nächster Nachbar als Start": _preset(start="nearest"),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps, 1 Million)": _preset(n=200, budget=1000000),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; Hill Climbing mit Neustarts bei gleichem Bewertungsbudget (der einzelne Abstieg ohne Budget)
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Tenure 20, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 4.19 % über der Schranke - ein Hill-Climbing-Abstieg 7.85 %, Hill Climbing mit Neustarts (voller Rescan) 4.88 %.",
    "Zu kleines Budget (25 Tausend)": "Nur 25 Tausend Vorschläge: 155.1 % über der Schranke - bei nur rund 15 Iterationen (jede bewertet die volle n²/2-Nachbarschaft) kommt die Suche von einer zufälligen Startlösung kaum voran. Ein einzelner Hill-Climbing-Abstieg braucht dieselbe Größenordnung Bewertungen, erreicht aber 7.9 %.",
    "Ohne Tabu (Tenure 0)": "Keine Kante bleibt gesperrt: 7.54 % über der Schranke, fast identisch mit einem einzelnen Hill-Climbing-Abstieg (7.85 %) - ohne Gedächtnis pendelt die Suche, sobald sie das erste lokale Optimum erreicht, nur noch zwischen zwei Touren.",
    "Zu lange Tenure (200)": "Kanten bleiben 200 Iterationen gesperrt (bei nur rund 114 Iterationen insgesamt praktisch für immer): 4.52 % über der Schranke, schlechter als die kalibrierte Tenure 20 (4.19 %) - zu viele sinnvolle Züge bleiben blockiert.",
    "Nächster Nachbar als Start": "Eine gute Startlösung hilft Tabu Search sichtbar (anders als bei ILS/VNS): 1.50 % über der Schranke gegen 4.19 % bei zufälliger Startlösung - bei nur rund 114 teuren Iterationen zählt jeder Startvorteil.",
    "Großes Budget (1 Million)": "1 Million Vorschläge, rund 566 Iterationen: 2.31 % über der Schranke - Hill Climbing mit Neustarts liegt bei 2.54 %, praktisch gleichauf.",
    "Große Instanz (200 Stopps, 1 Million)": "200 Stopps: jede Iteration kostet rund 20 Tausend Bewertungen (n²/2), 1 Million Vorschläge reichen nur für rund 50 Iterationen - die Suche kommt von einer zufälligen Startlösung praktisch nicht voran (307 % über der Schranke, gegen 9.5 % für Hill Climbing mit Neustarts).",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_hc", "comparable", "hc_wins"},
    "Zu kleines Budget (25 Tausend)": {"hc_wins"},
    "Ohne Tabu (Tenure 0)": {"hc_wins", "comparable", "beats_hc"},
    "Zu lange Tenure (200)": {"hc_wins", "comparable", "beats_hc"},
    "Nächster Nachbar als Start": {"beats_hc", "comparable", "hc_wins"},
    "Großes Budget (1 Million)": {"beats_hc", "comparable", "hc_wins"},
    "Große Instanz (200 Stopps, 1 Million)": {"hc_wins"},
}
