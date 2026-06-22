---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
<div style="float: right;">
  <a href="../de/bias_further.html" style="margin-left: 10px;">Deutsch</a>
  <a href="../en/bias_further.html">English</a>
</div>

# 3. Weitere Verzerrungen: Survivorship Bias und P-Hacking

Die vorherigen beiden Kapitel haben sich ausführlich mit der Verzerrung durch
ausgelassene Variablen und der Stichprobenverzerrung beschäftigt. Verzerrungen
treten jedoch in noch viel mehr Formen auf. Dieses Kapitel zeigt kurz zwei
weitere Verzerrungen, die in der Forschung häufig vorkommen. Beide Abschnitte
zeigen ein kompaktes, konkretes Beispiel, damit du das Muster in deiner eigenen
Arbeit wiedererkennen kannst.

```{code-cell}
:tags: ["remove_input", "remove_output"]
# Pakete installieren, wenn der Code in JupyterLite (Pyodide) über Thebe läuft.
try:
    import micropip
    await micropip.install(['scikit-learn', 'scipy', 'matplotlib', 'pandas', 'numpy'])
except ImportError:
    pass
```

```{code-cell}
:tags: ["remove_input", "remove_output"]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Hilfsfunktion ────────────────────────────────────────────────────────────
def rmse(y_true, y_pred):
    """Durchschnittlicher Vorhersagefehler in $/Std. (niedriger = besser)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ══════════════════════════════════════════════════════════════════════════
# Datensatz 1: Survivorship Bias
# ══════════════════════════════════════════════════════════════════════════
n1 = 3000
tenure_raw  = np.random.exponential(6, n1).clip(0.1, 30)
edu_s       = np.random.normal(14, 2.5, n1).clip(8, 20).round().astype(int)
exp_s       = np.random.normal(10, 5,   n1).clip(0, 30)
hours_s     = np.random.normal(40, 8,   n1).clip(20, 60)

wage_s = (
    5.0
    + 0.5  * edu_s
    + 0.3  * exp_s
    + 0.35 * tenure_raw
    + 0.08 * hours_s
    + np.random.normal(0, 2.5, n1)
).clip(10, None)

df_s = pd.DataFrame({
    'tenure':         tenure_raw,
    'education':      edu_s,
    'experience':     exp_s,
    'hours_per_week': hours_s,
    'wage':           wage_s,
})

# Arbeitnehmer mit kurzer Betriebszugehörigkeit UND unterdurchschnittlichem Lohn
# haben das Unternehmen wahrscheinlich verlassen: sie fanden einen besseren Job,
# wurden entlassen, oder befinden sich schlicht nicht mehr im Unternehmensarchiv.
wage_40th  = np.percentile(wage_s, 40)
leave_prob = np.where(
    (tenure_raw < 3) & (wage_s < wage_40th),
    0.80,   # 80% dieser Gruppe fehlen im Archiv
    0.08    # 8% Basisrate für alle anderen
)
survived = np.random.binomial(1, 1 - leave_prob, n1).astype(bool)

df_surv_full    = df_s.copy()            # die wahre Population
df_surv_archive = df_s[survived].copy()  # was das Archiv tatsächlich enthält

# ══════════════════════════════════════════════════════════════════════════
# Datensatz 2: P-Hacking
# ══════════════════════════════════════════════════════════════════════════
n2 = 5000
edu_p  = np.random.normal(14, 2.5, n2).clip(8, 20).round().astype(int)
exp_p  = np.random.normal(10, 5,   n2).clip(0, 30)

# 20 vollkommen zufällige Ja/Nein-Merkmale. Keines hat einen echten Einfluss auf den Lohn
FEATURE_NAMES = [
    'In ungeradem Monat geboren',  'Vorname < 5 Buchstaben',  'Besitzt ein Haustier',
    'Linkshänder',                 'Hat einen zweiten Vornamen', 'Am Wochenende geboren',
    'Fährt mit dem Auto zur Arbeit', 'Bevorzugt Tee statt Kaffee', 'Hat Geschwister',
    'Spielt ein Instrument',       'Nutzt öffentliche Verkehrsmittel', 'Hat einen Garten',
    'Liest Romane',                'Trägt eine Brille',          'Hat ein Tattoo',
    'Mag scharfes Essen',          'Frühaufsteher',              'Führt Tagebuch',
    'Bevorzugt Berge',             'Besitzt ein Fahrrad',
]
random_features = np.random.randint(0, 2, size=(n2, len(FEATURE_NAMES)))

wage_p = (
    5.0
    + 0.5 * edu_p
    + 0.3 * exp_p
    + 0.2 * random_features[:,-1]
    + np.random.normal(0, 3, n2)
).clip(10, None)
df_p = pd.DataFrame(random_features, columns=FEATURE_NAMES)
df_p['wage'] = wage_p

# ══════════════════════════════════════════════════════════════════════════
# Datensatz 3: Selection Bias (Bildung steuert die Teilnahme)
# ══════════════════════════════════════════════════════════════════════════
n3 = 1000
edu_sel = np.random.normal(13, 3, n3).clip(8, 20).round().astype(int)
exp_sel = np.random.normal(8,  5, n3).clip(0, 30)

# Hochgebildete Personen melden sich deutlich häufiger für das Programm an
raw_logit   = -1.5 + 0.20 * (edu_sel - 8)       # steigt mit der Bildung
enroll_prob = 1 / (1 + np.exp(-raw_logit))        # Sigmoid → hält Werte in (0, 1)
enrolled    = np.random.binomial(1, enroll_prob, n3).astype(bool)

TRUE_EFFECT = 0.50   # der wahre Effekt des Programms beträgt nur $0,50/Std.

wage_sel = (
    5.0
    + 0.70 * edu_sel
    + 0.25 * exp_sel
    + TRUE_EFFECT * enrolled          # kleiner, echter Effekt
    + np.random.normal(0, 2, n3)
).clip(10, None)

EDU_LABELS = ['Niedrig (≤12 Jahre)', 'Mittel (13–15 Jahre)', 'Hoch (≥16 Jahre)']
df_sel = pd.DataFrame({
    'education':  edu_sel,
    'experience': exp_sel,
    'enrolled':   enrolled.astype(int),
    'wage':       wage_sel,
})
df_sel['edu_group'] = pd.cut(
    df_sel['education'],
    bins=[0, 12, 15, 20],
    labels=EDU_LABELS
)
```

---

## Teil 1: Survivorship Bias

### Was ist das?

Stell dir vor, du untersuchst, was ein Restaurant erfolgreich macht, indem du nur
Restaurants besuchst, die *noch geöffnet* haben. Jedes Restaurant in deiner
Stichprobe hat überlebt, hat also bereits einen unsichtbaren Filter durchlaufen.
Die Restaurants, die gescheitert sind (und vielleicht viele Eigenschaften mit den
erfolgreichen geteilt hätten), sind schlicht nicht mehr da, um untersucht zu
werden. Deine Schlüsse über "Erfolgsfaktoren" werden vollständig durch diesen
unsichtbaren Filter geprägt.

Das ist **Survivorship Bias**: wenn deine Daten nur Datensätze von Einheiten
enthalten, die einen Auswahlprozess überstanden haben, während die
"Nicht-Überlebenden" stillschweigend ausgeschlossen werden. Das Ergebnis ist ein
Datensatz, der erfolgreicher, fähiger oder extremer wirkt als die tatsächliche
zugrunde liegende Population. Survivorship Bias ist damit ein Spezialfall der
Stichprobenverzerrung, der entsteht, wenn der Auswahlprozess selbst durch das
betrachtete Ergebnis beeinflusst wird.

### In einem Lohnarchiv

Unser synthetischer Datensatz simuliert ein Personalarchiv eines Unternehmens.
Arbeitnehmer mit kurzer Betriebszugehörigkeit **und** unterdurchschnittlichem
Lohn haben das Unternehmen weit häufiger bereits verlassen, weil sie anderswo
besser bezahlte Jobs gefunden haben oder entlassen wurden. Das Archiv enthält nur
die Arbeitnehmer, die geblieben sind. Die Grafik unten zeigt, wie dieser Filter
die Verteilung der Arbeitnehmer mit kurzer Betriebszugehörigkeit verzerrt.

```{code-cell}
:tags: ["remove_input"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for ax, df_plot, title in [
    (axes[0], df_surv_full,    'Wahre Population\n(alle Arbeitnehmer)'),
    (axes[1], df_surv_archive, 'Unternehmensarchiv\n(nur Überlebende)'),
]:
    n_short = (df_plot['tenure'] < 3).sum()
    n_long  = (df_plot['tenure'] >= 3).sum()
    ax.bar(['Kurze Zugehörigkeit\n(< 3 Jahre)', 'Lange Zugehörigkeit\n(3+ Jahre)'],
           [n_short, n_long],
           color=['mediumpurple', 'steelblue'], edgecolor='black', alpha=0.8)
    ax.set_ylabel('Anzahl Arbeitnehmer')
    ax.set_title(title)
    ax.set_ylim(0, (df_surv_full['tenure'] >= 3).sum() * 1.3)
    for i, v in enumerate([n_short, n_long]):
        ax.text(i, v + 20, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

Das Archiv enthält weit weniger Arbeitnehmer mit kurzer Betriebszugehörigkeit als
die wahre Population, weil die meisten Arbeitnehmer mit niedrigem Lohn und kurzer
Zugehörigkeit das Unternehmen bereits verlassen haben. Ein Modell, das nur mit
Archivdaten trainiert wird, hat diese Gruppe kaum gesehen und wird ihre Löhne
systematisch falsch vorhersagen.

```{code-cell}
feature_cols_s = ['tenure', 'education', 'experience', 'hours_per_week']

# Das Modell ausschließlich auf dem Archiv (den Überlebenden) trainieren
X_archive = df_surv_archive[feature_cols_s].values
y_archive  = df_surv_archive['wage'].values

# Auf der vollständigen Population testen, einschließlich der Arbeitnehmer, die gegangen wären
X_full = df_surv_full[feature_cols_s].values
y_full = df_surv_full['wage'].values
tenure_full_arr = df_surv_full['tenure'].values

lr_surv = LinearRegression()
lr_surv.fit(X_archive, y_archive)        # nur von den Überlebenden lernen
y_pred_surv = lr_surv.predict(X_full)    # für alle vorhersagen

short_mask_s = tenure_full_arr < 3
long_mask_s  = tenure_full_arr >= 3

print("Modell trainiert auf dem Archiv (nur Überlebende)")
print("=" * 60)
print(f"  Kurze Zugehörigkeit (< 3 Jahre): RMSE = "
      f"{rmse(y_full[short_mask_s], y_pred_surv[short_mask_s]):.2f} $/Std.  "
      f"(n = {short_mask_s.sum()})")
print(f"  Lange Zugehörigkeit  (3+ Jahre):  RMSE = "
      f"{rmse(y_full[long_mask_s],  y_pred_surv[long_mask_s]):.2f} $/Std.  "
      f"(n = {long_mask_s.sum()})")
print()
print("Die Gruppe mit kurzer Zugehörigkeit, die in den Trainingsdaten größtenteils")
print("fehlt, hat einen deutlich höheren Fehler. Das Modell hat die meisten ihrer")
print("Datensätze mit niedrigem Lohn nie gesehen.")
```

**Wichtige Erkenntnis:** Trainierst du ein Modell mit Daten, die bereits einen
Auswahlfilter durchlaufen haben (noch aktive Unternehmen, veröffentlichte
Studien, überlebende Projekte), wird das Modell bei den Fällen, die *nicht*
überlebt haben, schlecht abschneiden — oft genau bei den Fällen, bei denen du
verlässliche Vorhersagen am dringendsten brauchst.

---

## Teil 2: P-Hacking

### Was ist das?

Angenommen, du willst wissen, ob ein Glücksbringer die Prüfungsergebnisse
verbessert. Du testest 20 verschiedene Glücksbringer: eine Hasenpfote, ein
vierblättriges Kleeblatt, einen blauen Stift und so weiter. Selbst wenn keiner von
ihnen einen echten Effekt hat, wird etwa **1 von 20** Tests rein durch Zufall als
"statistisch signifikant" erscheinen (das ist die eigentliche Bedeutung eines
Signifikanzniveaus von 5 %). Veröffentlichst du dann *nur* dieses eine Ergebnis
"blaue Stifte verbessern Prüfungsergebnisse!", hast du **P-Hacking** betrieben:
das Herauspicken eines signifikanten Ergebnisses aus einer großen Menge von Tests,
während alle nicht signifikanten verworfen werden.

P-Hacking kann unabsichtlich passieren (eine Forschungsperson testet tatsächlich
viele Dinge und berichtet das "interessante" Ergebnis) oder absichtlich. In
beiden Fällen erhöht es die Zahl falscher Befunde in der wissenschaftlichen
Literatur.

### In einer Lohnstudie

Wir erzeugen Löhne, die nur von Bildung und Berufserfahrung abhängen. Dann
erfinden wir **20 vollkommen zufällige Ja/Nein-Merkmale** für jeden Arbeitnehmer:
Dinge wie "besitzt ein Haustier" oder "bevorzugt Berge", die absolut keinen
Zusammenhang mit dem Lohn haben. Anschließend testen wir jedes Merkmal mit einem
t-Test auf einen Lohneffekt (der fragt: "verdienen Arbeitnehmer mit diesem
Merkmal signifikant anders als jene ohne?").

```{code-cell}
# Für jedes Zufallsmerkmal einen t-Test durchführen: sagt es einen höheren Lohn voraus?
# (Keines dieser Merkmale hat einen echten Effekt, sie sind reines Rauschen.)
p_values = []
for feat in FEATURE_NAMES:
    wages_yes = df_p.loc[df_p[feat] == 1, 'wage']
    wages_no  = df_p.loc[df_p[feat] == 0, 'wage']
    _, p = stats.ttest_ind(wages_yes, wages_no)
    p_values.append(p)

results_p = (pd.DataFrame({'Merkmal': FEATURE_NAMES, 'p-Wert': p_values})
               .sort_values('p-Wert')
               .reset_index(drop=True))

print("t-Test-Ergebnisse für 20 zufällige Merkmale (keines ist ein echter Lohn-Prädiktor)")
print("=" * 60)
for _, row in results_p.iterrows():
    flag = "  ← SIGNIFIKANT (p < 0,05)" if row['p-Wert'] < 0.05 else ""
    print(f"  {row['Merkmal']:<32}: p = {row['p-Wert']:.3f}{flag}")

n_sig = (results_p['p-Wert'] < 0.05).sum()
print(f"\n{n_sig} von 20 Tests erscheinen bei p < 0,05 'signifikant'")
print(f"Rein durch Zufall erwartet: 20 × 0,05 = 1,0")
```

```{code-cell}
:tags: ["remove_input"]
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

# Links: horizontales Balkendiagramm aller p-Werte
bar_colors = ['tomato' if p < 0.05 else 'steelblue'
              for p in results_p['p-Wert']]
axes[0].barh(results_p['Merkmal'], results_p['p-Wert'],
             color=bar_colors, edgecolor='black', alpha=0.85)
axes[0].axvline(0.05, color='red', linestyle='--', linewidth=1.8,
                label='Schwelle p = 0,05')
axes[0].set_xlabel('p-Wert')
axes[0].set_title('p-Werte für 20 zufällige Merkmale\n(rote Balken = "signifikant")')
axes[0].legend()

# Rechts: Histogramm der p-Werte. Sollte unter der Nullhypothese annähernd gleichverteilt sein
axes[1].hist(p_values, bins=10, range=(0, 1),
             color='steelblue', edgecolor='black', alpha=0.8)
axes[1].axvline(0.05, color='red', linestyle='--', linewidth=1.8,
                label='p = 0,05')
axes[1].set_xlabel('p-Wert')
axes[1].set_ylabel('Anzahl Tests')
axes[1].set_title('Verteilung der p-Werte\n(gleichverteilt = kein echtes Signal in den Daten)')
axes[1].legend()

plt.tight_layout()
plt.show()
```

Das Histogramm rechts ist die entscheidende Diagnose: Wenn es wirklich keinen
Effekt gibt, sind die p-Werte annähernd gleichmäßig zwischen 0 und 1 verteilt. Nur
ein kleiner Häufungspunkt am linken Rand (unter 0,05) stellt die "falsch
positiven" Ergebnisse dar, die der Zufall erzeugt. Würde eine Forschungsperson nur
diese berichten, würde die veröffentlichte Literatur ein irreführendes Ergebnis
über Löhne enthalten.

P-Hacking kann viele verschiedene Formen annehmen und sowohl unabsichtlich als
auch absichtlich passieren. Zum Beispiel:
- Forschende könnten nicht-deterministische Algorithmen (wie Random Forests)
  mehrmals ausführen und nur das beste Ergebnis berichten.
- Sie könnten viele verschiedene Modellspezifikationen testen (Merkmale
  hinzufügen oder entfernen) und nur diejenige berichten, die ein signifikantes
  Ergebnis liefert.
- Sie könnten auch bestimmte Datenpunkte als "Ausreißer" betrachten und von der
  Analyse ausschließen, was die Ergebnisse beeinflussen kann.

**Wichtige Erkenntnis:** P-Hacking entsteht, wenn man durch zufälliges oder
gezieltes Herauspicken ein signifikantes Testergebnis erhält. Man kann die Daten,
das Modell manipulieren oder einfach so lange verschiedene Hypothesen testen, bis
ein signifikantes Ergebnis gefunden wird. Transparente Berichterstattung über
alle Tests und Korrekturen für multiples Testen (wie die Bonferroni-Korrektur)
sind die Standardmittel für eine gute wissenschaftliche Praxis.

## Zusammenfassung

| Verzerrung | Was schiefgeht | Wie man sie erkennt |
|---|---|---|
| **Survivorship** | Daten erfassen nur "Erfolge"; Fehlschläge sind unsichtbar | Frage: Wer ist *nicht* in diesem Datensatz, und warum? |
| **P-Hacking** | Tests werden manipuliert, um signifikante Ergebnisse zu erhalten | Berichte immer alle Ergebnisse und frage dich: War ich in dieser Analyse wirklich unvoreingenommen? |
