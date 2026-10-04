"""Tabu Search - eine Lieferrunde, die sich merkt, was sie gerade getan hat - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünftes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe: direktes Kind von Hill Climbing, wie die vier vorigen Stücke
auf derselben Lieferrunde. Anders als Simulated Annealing (Zufall), Iterated Local Search/Variable Neighborhood Search (gezielte
Störung + Wiederabstieg) tut Tabu Search etwas drittes: es bewertet wie Hill Climbing IMMER die volle Nachbarschaft und nimmt IMMER
den besten Zug - auch wenn er die Tour verlängert -, aber ein deterministisches Gedächtnis (die zuletzt entfernten Kanten sind
vorübergehend tabu) verhindert, dass der gerade gemachte Zug sofort wieder rückgängig gemacht wird. Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import tabu_constants as C
import tabu_tour as T
from tabu_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, scaling_table, sweep, verdict
from tabu_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from tabu_visualization import build_budget, build_instance, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Tabu Search – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


st.title("🚫 Tabu Search – eine Lieferrunde, die sich merkt, was sie gerade getan hat")
st.markdown(
    """
Wie **Hill Climbing** bewertet Tabu Search jede Iteration die **volle** Nachbarschaft und nimmt den besten Zug – aber anders als Hill Climbing hört es dabei nie auf: der beste Zug wird **immer** ausgeführt, auch wenn er die Tour verlängert. Damit die Suche dabei nicht sofort zwischen
zwei Touren hin- und herpendelt (den gerade gemachten Zug im nächsten Schritt einfach wieder rückgängig machen), werden die beiden **entfernten Kanten** für eine feste Zahl Iterationen **tabu**: ein Zug, der eine davon wiederherstellen würde, ist verboten – es sei denn, er verbessert die
beste je gefundene Tour (Aspirationskriterium). Kein Zufall, keine gezielte Störung – nur **Gedächtnis**. Der Preis: jede Iteration ist teuer (die volle Nachbarschaft, nicht nur ein Vorschlag), also passen viel weniger Iterationen ins Budget als bei den Geschwister-Demos.
"""
)
st.caption(
    "Fünftes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe: dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/), der "
    "[iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo) und der "
    "[variable-neighborhood-search-demo](https://github.com/sebastian-hanisch/variable-neighborhood-search-demo) - ein Depot in der Mitte, n Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen. "
    "GRASP und der Nachbarschafts-Zweig (Lin-Kernighan, VLSN, VRP-Nachbarschaften) sind andere Antworten auf dieselbe Schwäche der Wurzel."
)

with st.expander("So funktioniert Tabu Search", expanded=True):
    st.markdown(
        """
1. **Volle Nachbarschaft.** Jede Iteration wird die komplette 2-opt-Nachbarschaft bewertet (vektorisiert, wie Hill Climbings "beste Verbesserung") - rund n²/2 bewertete Nachbarn auf einen Schlag.
2. **Immer der beste Zug.** Anders als Hill Climbing wird IMMER der beste verfügbare Zug ausgeführt, auch wenn er die Tour verlängert - so kommt die Suche über ein lokales Optimum hinaus.
3. **Tabu-Liste.** Die zwei von einem Zug **entfernten** Kanten sind für `Tenure` Iterationen verboten (ein Zug, der eine davon wiederherstellen würde, ist tabu) - sonst würde der nächste Zug den eben gemachten sofort rückgängig machen.
4. **Aspiration.** Ein tabuer Zug wird trotzdem ausgeführt, wenn er die beste je gefundene Tour verbessert. Sind ausnahmsweise alle Züge tabu, wird der beste davon trotzdem ausgeführt (Notfall).
5. **Bewertung.** Der Abstand zur **1-Baum-Schranke**, wie in den Geschwister-Demos. Verglichen wird mit **einem Hill-Climbing-Abstieg** und **Hill Climbing mit Neustarts** (voller Rescan). Nur die Neustarts bekommen dasselbe Bewertungsbudget wie Tabu Search; der einzelne Abstieg läuft ohne Budget bis zum lokalen Optimum (im Mittel rund 100 Tausend Bewertungen bei 60 Stopps).
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:4], preset_names[4:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Jede Iteration kostet rund n²/2 Bewertungen - bei 200 Stopps schon rund 20 Tausend je Iteration.",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    tenure = st.slider(
        "Tabu-Tenure", C.TENURE_MIN, C.TENURE_MAX, key="tenure_slider",
        help="Wie viele Iterationen eine entfernte Kante verboten bleibt. Bei 200 Tausend Vorschlägen: 0 (kein Gedächtnis) 7.54 %, 5 → 6.12 %, 10 → 5.18 %, 20 → **4.19 %** (Voreinstellung, bestes gemessen), 50 → 4.52 %, 200 → 4.52 % über der Schranke - "
             "ein Sweet Spot, nicht 'länger ist sicherer': bei zu hoher Tenure bleiben zu viele sinnvolle Züge blockiert.",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarn)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Bei 10 / 25 / 50 / 100 / 200 Tausend / 0.5 / 1 / 2 Millionen liegt die beste Tour **287.8 / 155.1 / 54.7 / 7.9 / 4.2 / 3.1 / 2.3 / 1.9 %** über der Schranke. "
             "Jede Iteration bewertet die volle Nachbarschaft (~1800 Bewertungen bei 60 Stopps) statt eines einzelnen Vorschlags - unter 100 Tausend reichen die paar Dutzend Iterationen nicht, um von einer zufälligen Startlösung wegzukommen. "
             "Erst ab 200 Tausend schlägt Tabu Search Hill Climbing mit Neustarts (voller Rescan) klar; bei 2 Millionen liegen beide gleichauf.",
    )
    start = st.radio(
        "Startlösung", list(C.START_LABELS), key="start_radio", format_func=lambda k: C.START_LABELS[k], horizontal=True,
        help="Zufällige Reihenfolge oder Nächster Nachbar. Anders als bei ILS/VNS zählt eine gute Startlösung hier sichtbar: bei nur rund 114 teuren Iterationen (200 Tausend Vorschläge) ist ein Vorsprung schwerer aufzuholen (1.50 % gegen 4.19 %).",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert nur die zufällige Startlösung - Tabu Search selbst ist deterministisch (immer der beste erlaubte Zug, kein Zufall).",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "tenure_slider": int(tenure),
    "budget_select": int(budget), "start_radio": start, "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), int(tenure), int(budget), start, int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
run = a.run
xy = a.inst.xy
code = verdict(a)
data_key = settings

# --- Tabu Search in Aktion ---------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Tabu Search in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Suche", 3: "3 · Ergebnis"}
if "tabu_step" not in st.session_state or st.session_state.get("tabu_step_owner") != data_key:
    st.session_state["tabu_step"] = 1
    st.session_state["tabu_step_owner"] = data_key
    st.session_state.pop("tabu_iter", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="tabu_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.snapshots)
iteration = n_snaps - 1
play_search = False
if step == 2 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        iteration = st.slider("Iteration", 0, n_snaps - 1, value=n_snaps - 1, key="tabu_iter", help="Die Tour nach dieser Iteration (0 = Startlösung, vor der ersten Iteration).")
    with itplay_col:
        play_search = st.button("▶️ Suche abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{a.inst.n} Kundenstopps und das Depot (Stern)** – {a.inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            st.markdown("**Länge der aktuellen und der besten Tour über die bewerteten Nachbarn**")
            st.plotly_chart(build_trace(run.trace_iter, run.trace_length, run.trace_best, a.bound, a.hc.length, T.tour_length(a.hcr_tour, a.D)), width="stretch", key=f"s2_trace_{it}")
            st.markdown(f"**Tour nach Iteration {it} von {n_snaps - 1}**")
            st.plotly_chart(build_tour(xy, run.snapshots[it], ghost=run.best_tour if it < n_snaps - 1 else None), width="stretch", key=f"s2_map_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**Tabu Search: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, run.best_tour), width="stretch", key="s3_tabu")
            c2.markdown(f"**Hill Climbing mit Neustarts** ({a.hcr_starts} Abstiege, gleiches Budget) – {a.hcr_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.hcr_tour), width="stretch", key="s3_hc")


if auto_play:
    for s in STEP_LABELS:
        if s == 2:
            for f in _frames():
                _render(2, f)
                time.sleep(0.1)
            time.sleep(0.6)
        else:
            _render(s, iteration)
            time.sleep(1.2)
    step = 3
elif play_search:
    for f in _frames():
        _render(2, f)
        time.sleep(0.1)
else:
    _render(step, iteration)

if step == 1:
    st.caption(f"{a.inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {a.bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Nachbarn in {run.iterations} Iterationen (rund {run.evaluations // max(run.iterations, 1)} je Iteration - die volle Nachbarschaft); {run.overrides} davon mit Notfall-Überschreibung (alle Züge tabu).")
else:
    st.caption(f"Links die beste Tour aus {run.iterations} Iterationen, rechts die beste Tour aus {a.hcr_starts} Hill-Climbing-Abstiegen mit demselben Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis -------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung (nur die Startlösung streut, Tabu Search selbst ist deterministisch): Vergleiche gelten für diesen Lauf."
)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Beste Tour", f"{a.gap:.1f} %", delta=f"letzte Tour {a.final_gap:.1f} %", delta_color="off", help="Abstand zur Schranke der kürzesten je besuchten Tour; im Delta der der letzten Tour der Kette.")
m2.metric("Ein Hill-Climbing-Abstieg", f"{a.hc_gap:.1f} %", delta=f"{_fmt_int(a.hc.evaluations)} Bewertungen", delta_color="off", help="Ein Abstieg (beste Verbesserung, dieselbe Zugauswahl wie Tabu Search ohne Gedächtnis) aus derselben Startlösung, ohne Budget bis zum lokalen Optimum - die angezeigten Bewertungen sind das, was er dafür braucht.")
m3.metric("HC mit Neustarts (voller Rescan)", f"{a.hcr_gap:.1f} %", delta=f"{a.hcr_starts} Abstiege, gleiches Budget", delta_color="off", help="So viele Abstiege aus zufälligen Startlösungen, wie ins Budget passen; die beste Tour zählt.")
m4.metric("Notfall-Überschreibungen", f"{a.override_rate:.0%}", delta=f"{run.overrides} von {run.iterations}", delta_color="off", help="Anteil der Iterationen, in denen ALLE gültigen Züge tabu waren (zu hohe Tenure für diese Instanz).")

if code == "beats_hc":
    st.success(f"✅ Besser als Hill Climbing mit Neustarts (voller Rescan, gleiches Budget): {a.gap:.1f} % über der Schranke gegen {a.hcr_gap:.1f} % ({a.hcr_starts} Abstiege). Das Gedächtnis erlaubt es der Suche, über lokale Optima hinauszukommen, ohne Struktur wegzuwerfen. Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: Tabu Search {a.gap:.1f} %, Hill Climbing mit Neustarts {a.hcr_gap:.1f} % über der Schranke. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ Hill Climbing mit Neustarts ist besser: {a.hcr_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget ({a.hcr_starts} Abstiege). Bei knappem Budget reichen die wenigen, teuren Iterationen von Tabu Search nicht - siehe README 'Was nicht funktioniert hat'.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarn", "Rechenzeit"],
              "Tabu Search (beste Tour)": [f"{run.best_length:.1f}", f"{a.gap:.2f} %", _fmt_int(run.evaluations), unit_time(a.seconds)],
              "Tabu Search (letzte Tour)": [f"{run.final_length:.1f}", f"{a.final_gap:.2f} %", "–", "–"],
              "Ein Abstieg": [f"{a.hc.length:.1f}", f"{a.hc_gap:.2f} %", _fmt_int(a.hc.evaluations), unit_time(a.hc_seconds)],
              "HC mit Neustarts": [f"{T.tour_length(a.hcr_tour, a.D):.1f}", f"{a.hcr_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.hcr_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Tabu-Tenure", "Budget", "Iterationen", "Bewertungen je Iteration", "Startlösung", "Kreuzungen der besten Tour"],
              "Einstellung": [f"{settings.tenure}", _fmt_int(settings.budget), f"{run.iterations}", f"{run.evaluations // max(run.iterations, 1)}", C.START_LABELS[settings.start], f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist ein bewerteter Nachbar, dieselbe Einheit wie in den Geschwister-Demos - aber Tabu Search bewertet die volle Nachbarschaft auf einen Schlag, nicht einzelne Vorschläge nacheinander. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps -----------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget, Tenure und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 20 bis 90 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    categorical = sweep_param == "start"
    labels = {"start": C.START_LABELS}.get(sweep_param)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param], categorical=categorical, key_labels=labels), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band bzw. Balken) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gestrichelt: ein Hill-Climbing-Abstieg; gepunktet: Hill Climbing mit Neustarts (voller Rescan).")

st.markdown("---")

# --- Experimente ------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Budget: wie teuer ist eine Iteration wirklich?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 90 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "Tabu Search (%)": [f"{r['gap']:.2f}" for r in rows_b],
              "HC-Neustarts (%)": [f"{r['hcr']:.2f}" for r in rows_b], "ein Abstieg (%)": [f"{r['hc']:.2f}" for r in rows_b], "Iterationen": [f"{r['iterations']:.0f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, Tenure 20). Bei 10-50 Tausend Vorschlägen reichen nur 6-29 Iterationen (jede bewertet die volle n²/2-Nachbarschaft) - katastrophal weit von der Schranke entfernt (**55-288 %**). "
               "Erst ab 100 Tausend erreicht Tabu Search die Güte eines einzelnen Hill-Climbing-Abstiegs (7.9 % gegen 7.85 %); erst ab 200 Tausend schlägt es Hill Climbing mit Neustarts klar (4.2 % gegen 4.9 %). Bei 2 Millionen liegen beide gleichauf (1.9 %). "
               "Kein anderes Stück dieser Linie braucht ein derart hohes Mindestbudget, um überhaupt konkurrenzfähig zu werden - der Preis für garantiert den besten Zug je Iteration.")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 15 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten und 20 Abstiege..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["tabu"], sp["hc"]), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric("Tabu Search: Mittel ± Streuung", f"{sp['tabu'].mean():.2f} ± {sp['tabu'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten (nur die Startlösung streut).")
    s2.metric("Ein Hill-Climbing-Abstieg: Mittel ± Streuung", f"{sp['hc'].mean():.2f} ± {sp['hc'].std():.2f} %", help="Ein Abstieg je Kette aus derselben zufälligen Startlösung.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (nur die Startlösung wechselt - Tabu Search selbst ist deterministisch).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 90 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget). Eine Iteration kostet rund n²/2 Bewertungen - bei 200 Stopps schon rund 20 Tausend, bei 1 Million Vorschlägen also nur rund 50 Iterationen: "
               "**307 %** über der Schranke, weit hinter Hill Climbing mit Neustarts (9.5 %). Tabu Search skaliert deutlich schlechter mit der Instanzgröße als die Kandidatenlisten-Familie (ILS, VNS).")

st.markdown("---")

# --- Grenzen ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Budget reicht für genug teure Iterationen** | Bei 25 Tausend Vorschlägen (60 Stopps, nur ~15 Iterationen): **155 %** über der Schranke - katastrophal. Erst ab 100 Tausend erreicht Tabu Search die Güte eines einzelnen Hill-Climbing-Abstiegs, erst ab 200 Tausend schlägt es Hill Climbing mit Neustarts. | **ILS/VNS** (viele billige Iterationen statt weniger teurer) |
| **Die Tenure passt zur Instanz** | Ohne Gedächtnis (Tenure 0): **7.54 %**, kaum besser als ein Abstieg (7.85 %) - die Suche pendelt, sobald sie ein lokales Optimum erreicht. Zu lange Tenure (200 bei nur ~114 Iterationen): **4.52 %**, schlechter als die kalibrierte Tenure 20 (4.19 %) - zu viele Züge bleiben blockiert. | Kein direkter Nachfolger; dieselbe Lehre wie SA's Temperatur, ILS' Störstärke, VNS' k_max |
| **Jede Iteration ist bezahlbar** | Bei 200 Stopps kostet eine Iteration rund 20 Tausend Bewertungen - 1 Million Vorschläge reichen nur für ~50 Iterationen (**307 %** über der Schranke). Skaliert deutlich schlechter als die Kandidatenlisten-Familie. | **Kandidatenliste + Don't-Look-Bits** (aus der Hill-Climbing-Demo: dieselbe Idee, nur die geänderte Umgebung neu bewerten, statt immer die volle Nachbarschaft) |
| **Die Startlösung zählt weniger als das Gedächtnis** | Anders als bei ILS/VNS: Nächster Nachbar (1.50 %) schlägt eine zufällige Startlösung (4.19 %) deutlich - bei nur rund 114 teuren Iterationen zählt jeder Vorsprung. | (kein Nachfolger nötig - eine gute Konstruktionsheuristik hilft überall, aber hier besonders) |
"""
)
st.caption(
    "Die Nachbarn der Trajektorien-Metaheuristiken-Linie: GRASP (randomisierte Konstruktion, viele Starts) und der Nachbarschafts-Zweig "
    "(Lin-Kernighan, VLSN, VRP-Nachbarschaften) sind andere Antworten auf dieselbe Schwäche der Wurzel; ALNS braucht eine CVRP-Instanz und ist deshalb hier noch nicht gebaut."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**Zug.** Ein 2-opt-Zug $(i, j)$ entfernt die Kanten $(t_i, t_{i+1})$ und $(t_j, t_{j+1})$, fügt $(t_i, t_j)$ und $(t_{i+1}, t_{j+1})$ ein (Stück zwischen $i+1$ und $j$ umgekehrt).

**Tabu-Liste.** $\text{tabu}(u, v)$ = die Iteration, bis zu der die Kante $\{u, v\}$ verboten ist. Nach einem Zug, der die Kanten $e_1, e_2$ entfernt: $\text{tabu}(e_1) \leftarrow \text{tabu}(e_2) \leftarrow \text{aktuelle Iteration} + \text{Tenure}$.

**Zugauswahl.** In jeder Iteration: $\pi' = \arg\min_{(i,j) \text{ gültig}} L(\pi \text{ mit Zug } (i,j))$ unter den Zügen, deren neue Kanten nicht tabu sind - **es sei denn**, der Zug verbessert die beste je gefundene Tour (Aspiration: $L(\pi') < L(\pi_{\text{best}})$), dann ist er trotzdem erlaubt. Sind alle gültigen Züge tabu, wird der global beste trotzdem ausgeführt (Notfall).
Anders als bei Hill Climbing wird der Zug **immer** ausgeführt, auch wenn $L(\pi') > L(\pi)$.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgrößen: ein Hill-Climbing-Abstieg (beste Verbesserung, dieselbe Zugauswahl ohne Gedächtnis; ohne Budget, bis zum lokalen Optimum) und Hill Climbing mit Neustarts (voller Rescan, wie in der Wurzel-Demo; bei gleichem Bewertungsbudget wie Tabu Search).

**Grenzen.** (1) Eine Iteration kostet $O(n^2)$ Bewertungen - das nötige Mindestbudget ist viel höher als bei einer Kandidatenliste. (2) Die Tenure ist ein Sweet-Spot-Parameter, kein "länger ist sicherer". (3) Ohne Tabu-Mechanismus pendelt die Suche nach dem ersten lokalen Optimum.

Implementiert in `tabu_algorithm.py` (die Suchschleife: volle Nachbarschaft, Tabu-Maske, Aspiration, Notfall-Überschreibung), `tabu_tour.py` (Nachbarschaften, Abstieg, Schranke - aus der Hill-Climbing-Demo), `tabu_scenario.py` (Instanzen), `tabu_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Trajektorien-Metaheuristiken: HC bis ALNS](https://sebastianhanisch.net/konzepte-trajektorien-metaheuristiken.html)."
)
