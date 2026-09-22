# Tabu Search – eine Lieferrunde, die sich merkt, was sie gerade getan hat – Streamlit-Demo

**[→ Demo live ausprobieren](#)** (Deploy offen)

Fünftes Stück der **Trajektorien-Metaheuristiken-Linie** der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning":
dieselbe Rundtour wie in der [hill-climbing-demo](../hill-climbing-demo), der [simulated-annealing-demo](../simulated-annealing-demo), der [iterated-local-search-demo](../iterated-local-search-demo) und der [variable-neighborhood-search-demo](../variable-neighborhood-search-demo) (ein Depot, n Kundenstopps in einem 100 × 100-km-Gebiet), dieselbe untere Schranke.

**Einordnung in die Reihe:** **Tabu Search** (Glover 1986, formalisiert 1989/90) ist ein direktes Kind von Hill Climbing, wie die vier vorigen Stücke - aber mit einem grundlegend anderen Rechenprofil. ILS und VNS bewerten je Iteration nur eine **Kandidatenliste** rund um eine gezielte Störung (billig, viele Iterationen passen ins Budget). Tabu Search bewertet wie Hill Climbing selbst jede Iteration die **volle** Nachbarschaft (teuer, wenige Iterationen passen ins Budget) - nimmt aber immer den besten Zug, auch wenn er die Tour verlängert, und verhindert mit einem **deterministischen Gedächtnis** (die zuletzt entfernten Kanten sind vorübergehend tabu), dass dieser Zug sofort wieder rückgängig gemacht wird.
```
hill-climbing-demo (Wurzel: nur bergab, bleibt im ersten Optimum stecken)        [gebaut]
  ├─ simulated-annealing-demo (nimmt Verschlechterungen an, Abkühlplan)          [gebaut]
  ├─ iterated-local-search-demo (stört ein gutes Optimum mit fester Störstärke)  [gebaut]
  │     └─ variable-neighborhood-search-demo (Störstärke eskaliert + Reset)     [gebaut]
  ├─ tabu-search-demo (immer der beste Zug, Gedächtnis gegen Rückwege)          [dieses Stück]
  └─ GRASP                    (randomisierte Konstruktion, viele Starts)         [nicht gebaut]
```

Ergebnis in Kürze: **Tabu Search braucht ein deutlich höheres Mindestbudget als jedes andere Stück dieser Linie, um überhaupt konkurrenzfähig zu werden – aber danach lohnt sich das Gedächtnis.** 60 Stopps: bei 10-50 Tausend Vorschlägen (nur 6-29 Iterationen, jede bewertet die volle n²/2-Nachbarschaft) liegt die beste Tour **katastrophal** weit von der Schranke entfernt (55-288 %). Erst ab **100 Tausend** erreicht Tabu Search die Güte eines einzelnen Hill-Climbing-Abstiegs (7.9 % gegen 7.85 %), erst ab **200 Tausend** schlägt es Hill Climbing mit Neustarts (voller Rescan) klar (4.19 % gegen 4.88 %). Bei 2 Millionen Vorschlägen liegen beide wieder gleichauf (1.94 % gegen 1.87 %) – der Vorsprung ist ein Fenster, kein Dauerzustand.
Die **Tabu-Tenure** (wie lange eine entfernte Kante gesperrt bleibt) hat einen klaren Sweet Spot bei 20 - ohne Gedächtnis (Tenure 0) landet die Suche fast exakt bei einem einzelnen Hill-Climbing-Abstieg (7.54 % gegen 7.85 %, sie pendelt nach dem ersten lokalen Optimum nur noch zwischen zwei Touren), bei zu langer Tenure (200) bleiben zu viele sinnvolle Züge blockiert (4.52 % statt 4.19 %) - dieselbe "Sweet Spot, nicht länger ist besser"-Lehre wie bei SA, ILS und VNS.

| Frage | Ergebnis (60 gleichverteilte Stopps, Tenure 20, zufällige Startlösung; Mittel über 5 feste Instanzen, Seeds 100000–100004, mit je 3 Ketten-Seeds; Abstand = Prozent über der 1-Baum-Schranke) |
|---|---|
| Standardfall | ➖ Tabu Search **4.19 %** über der Schranke gegen **7.85 %** für einen Hill-Climbing-Abstieg und **4.88 %** für Hill Climbing mit Neustarts (voller Rescan) |
| **Budget** | ❌❗ Bei 10 / 25 / 50 Tausend Vorschlägen: **287.8 / 155.1 / 54.7 %** über der Schranke (katastrophal - nur 6-29 Iterationen). Bei 100 Tausend: **7.9 %** (≈ ein Abstieg). Bei 200 / 500 Tausend / 1 / 2 Millionen: **4.2 / 3.1 / 2.3 / 1.9 %** - ab hier klar besser als Neustarts, bei 2 Millionen wieder gleichauf |
| **Tabu-Tenure** | ⚠️ 0 / 5 / 10 / 20 / 50 / 200: **7.54 / 6.12 / 5.18 / 4.19 / 4.52 / 4.52 %** - ein Sweet Spot bei 20, kein "länger ist sicherer" |
| **Ohne Tabu (Tenure 0)** | ❌ **7.54 %**, fast identisch mit einem einzelnen Hill-Climbing-Abstieg (7.85 %) - die Suche pendelt nach dem ersten lokalen Optimum zwischen zwei Touren, ohne Fortschritt |
| **Größe** | ❌❗ 200 Stopps, 1 Million Vorschläge (nur ~50 Iterationen, jede kostet ~20 Tausend Bewertungen): **318 %** über der Schranke gegen 10.7 % für Hill Climbing mit Neustarts - skaliert deutlich schlechter als die Kandidatenlisten-Familie |
| **Startlösung** | ✅ Nächster Nachbar **0.64 %** gegen zufällig **3.74 %** - anders als bei ILS/VNS zählt die Startlösung hier deutlich (bei nur ~114 teuren Iterationen ist ein Vorsprung schwer aufzuholen) |

## Was die Demo zeigt

1. **Tabu Search in Aktion** (Schritt-Slider + Abspielen): **Instanz** → **Suche** (Iterations-Regler + ▶️ Suche abspielen: Länge der aktuellen/besten Tour über die bewerteten Nachbarn, dazu die Tour nach der gewählten Iteration) → **Ergebnis** (beste Tour neben der besten aus Hill Climbing mit Neustarts).
2. **Was die Suche gefunden hat:** beste und letzte Tour, ein Abstieg, Hill Climbing mit Neustarts, Notfall-Überschreibungen; Urteil (`beats_hc` → `comparable` → `hc_wins`), Detailtabellen.
3. **📐 Sweeps** über Budget, Tenure, Stopps, Gruppen und Startlösung (feste Instanzen ab 100000, drei Ketten je Instanz).
4. **🔬 Experimente auf Abruf:** Budget von 10 Tausend bis 2 Millionen (die "teure Iteration"-Geschichte); Streuung über 20 Ketten; Skalierung von 20 bis 200 Stopps.
5. **🚧 Grenzen:** Tabelle "Annahme – was passiert – wer setzt an" (das Budget reicht für genug teure Iterationen, die Tenure passt zur Instanz, jede Iteration ist bezahlbar, die Startlösung zählt).

Regler: Stopps (10–200), Anteil der Stopps in Gruppen, **Tabu-Tenure** (0–200), **Budget** (10 Tausend bis 2 Millionen bewertete Nachbarn), **Startlösung**, Seed der Instanz (+ 🎲), Seed der Kette (+ 🎲).
Kein Nachbarschafts-Regler (bewusst nur 2-opt - das Tabu-Attribut ist an Kanten geknüpft, wie VNS' Verzicht auf VND eine explizite Scope-Entscheidung); kein Zufalls-Seed für die Suche selbst (Tabu Search ist deterministisch, nur die Startlösung streut).

## Messwerte der Presets (Instanz-Seed 35, Ketten-Seed 0; sie prüfen sich mit Urteil-Bändern selbst)

| Preset | Urteil (Band über Instanzen × Ketten) |
|---|---|
| Standardfall (Voreinstellung) | beats_hc / comparable / hc_wins |
| Zu kleines Budget (25 Tausend) | hc_wins |
| Ohne Tabu (Tenure 0) | beats_hc / comparable / hc_wins |
| Zu lange Tenure (200) | beats_hc / comparable / hc_wins |
| Nächster Nachbar als Start | beats_hc / comparable / hc_wins |
| Großes Budget (1 Million) | beats_hc / comparable / hc_wins |
| Große Instanz (200 Stopps, 1 Million) | hc_wins |

## Modell und Verfahren

- **Instanz, Nachbarschaften, Abstieg, Schranke** (`tabu_scenario.py`, `tabu_tour.py`): wortgleiche Kopien aus der [hill-climbing-demo](../hill-climbing-demo) (per Test gegen eingefrorene Werte) - Tabu Search braucht keine Kandidatenliste/Doppelbrücke der ILS/VNS-Familie, sondern direkt die volle, vektorisierte 2-opt-Bewertung der Wurzel-Demo.
- **Tabu-Suchschleife** (`tabu_algorithm.py`): jede Iteration wird die volle 2-opt-Nachbarschaft bewertet (vektorisiert); die zwei von einem Zug entfernten Kanten sind für `Tenure` Iterationen tabu (ein Zug, der eine davon wiederherstellt, ist verboten); Aspirationskriterium überschreibt Tabu bei einem neuen Bestwert; Notfall-Überschreibung, falls ausnahmsweise alle Züge tabu sind. Immer der beste erlaubte Zug wird ausgeführt, auch wenn er die Tour verlängert. Deterministisch - kein Zufall im Kern.
- **Hill Climbing mit Neustarts** (`tabu_evaluation.py`): Abstiege aus zufälligen Startlösungen, voller Rescan (wie in der Wurzel-Demo), bis das Budget erreicht ist. Kein Kandidatenlisten-Vergleich wie bei ILS/VNS: Tabu Search ist selbst schon "voller Rescan"-nativ, ein DLB-Vergleich wäre keine faire Gegenüberstellung (siehe [[hill-climbing-demo]] für den DLB-Fund).
- **Auswertung** (`tabu_evaluation.py`): Kennzahlen, Urteil, Sweeps über feste Instanzen × Ketten, Streuung, Skalierung.

## Was nicht funktioniert hat / Grenzen

- **Vorab-Vermutung: "deterministisches Gedächtnis schlägt zufällige/perturbative Verfahren bei gleichem Budget"** – **weder klar bestätigt noch widerlegt, sondern budgetabhängig**: bei knappem Budget (unter 100 Tausend) verliert Tabu Search KATASTROPHAL, nicht nur knapp - die volle Nachbarschaftsbewertung (~n²/2 Bewertungen je Iteration) lässt bei 10-50 Tausend Vorschlägen nur 6-29 Iterationen zu, zu wenig für irgendeinen echten Fortschritt von einer zufälligen Startlösung (55-288 % über der Schranke). Das ist die deutlichste "teure Iterationen"-Geschichte der ganzen Linie: kein anderes Stück (SA, ILS, VNS) hat ein derart hohes Mindestbudget, unter dem es komplett unbrauchbar wird.
  Erst ab 100 Tausend Vorschlägen erreicht Tabu Search die Güte eines einzelnen Hill-Climbing-Abstiegs, erst ab 200 Tausend schlägt es Hill Climbing mit Neustarts klar - und bei sehr großem Budget (2 Millionen) nähert sich der Vorsprung wieder auf null an. Das Fazit ist also nicht "Gedächtnis gewinnt", sondern "Gedächtnis gewinnt in einem mittleren Budgetfenster, wenn man sich seine teuren Iterationen leisten kann".
- **Die Tabu-Tenure ist ein Sweet-Spot-Parameter**, nicht "länger ist sicherer" - Tenure 20 schlägt sowohl Tenure 0 (kein Gedächtnis) als auch Tenure 200 (zu lange gesperrt). Dieselbe Lehre wie SA's Temperatur, ILS' Störstärke und VNS' k_max, hier zum fünften Mal bestätigt.
- **Eine gute Startlösung hilft deutlich** (Nächster Nachbar 0.64 % gegen zufällig 3.74 %) - ein auffälliger Kontrast zu ILS/VNS, wo die Startlösung praktisch irrelevant war (die vielen billigen Iterationen dort vergessen sie schnell; die wenigen teuren Iterationen hier können das nicht).
- **Skaliert schlechter mit der Instanzgröße** als die Kandidatenlisten-Familie: bei 200 Stopps kostet eine Iteration rund 20 Tausend Bewertungen, 1 Million Vorschläge reichen nur für rund 50 Iterationen (318 % über der Schranke) - der naheliegende nächste Schritt wäre eine Kandidatenlisten-Variante von Tabu Search, hier bewusst nicht gebaut (siehe Grenzen-Tabelle).
- **Synthetische Instanzen:** euklidisch, gleichverteilt oder in fünf Gruppen, ein Fahrzeug, keine Kapazitäten oder Zeitfenster. Zeiten hängen vom Rechner und der Python-Version ab (die Tests prüfen nur Größenordnungen).

## Verifikation

- **Tabu-Kanten-Verwaltung:** unabhängige Nachrechnung (aus Momentaufnahmen + `debug_trace`) baut die Tabu-Liste NUR anhand der öffentlichen Regel neu auf (nicht anhand der internen Datenstruktur des Algorithmus) und bestätigt: der gewählte Zug ist bei jeder Iteration ohne Notfall-Überschreibung der beste unter den so rekonstruierten erlaubten Zügen; eine entfernte Kante ist genau `Tenure` Iterationen gesperrt; Aspiration überschreibt Tabu nachweislich nur bei einem neuen Bestwert.
- **"Ohne Tabu"-Grenzfall (Tenure 0):** die Suche pendelt nachweislich zwischen genau zwei Touren, sobald sie ein lokales Optimum erreicht (keine Kante bleibt gesperrt, der letzte Zug ist sofort wieder der beste).
- **Notfall-Überschreibung:** bei einer Tenure, die größer als die Instanz ist, muss sie zwangsläufig greifen (sonst bliebe die Suche stecken) - bei Tenure 0 greift sie nie.
- Übernommener Kern: 2-opt gegen Brute-Force, Abstieg strikt monoton und im lokalen Optimum, Bewertungsbudget, 1-Baum-Schranke gegen Brute-Force (n = 8) und CP-SAT (n = 20); Instanz gegen eingefrorene Werte.
- **Alle Zahlen der App-Texte sind als Tests hinterlegt** (Seitenleiste, Presets, Grenzen-Tabelle, Budget-, Tenure- und Größen-Aussagen; jeweils Mittel über die festen Sweep-Instanzen × Ketten; positive **und** negative Aussagen; Rechenzeiten nur als Größenordnung); alle 7 Presets über mehrere Instanzen und Ketten in Urteil-Bändern; AppTest-Rauchtests (Voreinstellung, jedes Preset, jeder Schritt bei 10 und 60 Stopps, Iterations-Regler, ▶️ Abspielen und ▶️ Suche abspielen ohne doppelte Schlüssel, Würfel-Knöpfe, Permalink-Grenzen, Extremwerte, Experimente auf Abruf, Footer).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App: Schritte, Ergebnis, 📐 Sweeps, 🔬 Experimente (Budget, Streuung, Skalierung), 🚧 Grenzen, Mathe |
| `tabu_algorithm.py` | Die Tabu-Suchschleife: volle Nachbarschaft, Tabu-Maske, Aspiration, Notfall-Überschreibung |
| `tabu_tour.py` | Nachbarschaften, Abstieg (mit Bewertungsbudget), Kreuzungen, 1-Baum-Schranke (aus der Hill-Climbing-Demo) |
| `tabu_scenario.py`, `tabu_constants.py` | Instanzen; Konstanten, Presets |
| `tabu_evaluation.py` | Analyse, Urteil, Hill Climbing mit Neustarts, Sweeps, Streuung, Skalierung |
| `tabu_presets.py`, `tabu_visualization.py` | Permalink/Presets, Plotly-Figuren (achsengesperrt) |
| `tests/` | Übernommener Kern, Tabu-Suchschleife (unabhängige Replay-Verifikation), Szenario und Auswertung, Aussagen der App, Presets, AppTest |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
