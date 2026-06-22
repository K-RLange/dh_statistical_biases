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
  <a href="../de/bias_discussion.html" style="margin-left: 10px;">Deutsch</a>
  <a href="../en/bias_discussion.html">English</a>
</div>

# 4. Diskussion und Quiz

## Was wir gesehen haben: Eine Zusammenfassung

Die vorherigen drei Kapitel haben jeweils eine andere Art gezeigt, wie ein Modell
nicht durch einen Programmierfehler, sondern durch einen Mangel in den Daten oder
im Analyseprozess danebenliegen kann. Die folgende Tabelle fasst die vier
behandelten Arten von Verzerrungen zusammen.

| Verzerrung | Kapitel | Was schiefgeht | Charakteristisches Symptom |
|---|---|---|---|
| **Ausgelassene Variable** | 1 | Ein wichtiger Prädiktor fehlt im Modell | Systematische Über- oder Unterschätzung für bestimmte Gruppen |
| **Stichprobe** | 2 | Trainingsdaten repräsentieren nicht die Zielpopulation | Große Fehler für unterrepräsentierte Gruppen zur Testzeit |
| **Survivorship** | 3 | Nur "erfolgreiche" Fälle erscheinen in den Daten; Fehlschläge sind unsichtbar | Modell performt gut bei Überlebenden, aber schlecht in der Gesamtpopulation |
| **P-Hacking** | 3 | Tests werden wiederholt, bis ein signifikantes Ergebnis erscheint | Veröffentlichte Befunde, die sich nicht replizieren lassen |

Obwohl sie auf unterschiedliche Weise entstehen, haben alle vier eine gemeinsame
Wurzel: Die Daten oder die Analyse bilden die Welt, die das Modell beschreiben
soll, nicht getreu ab.

---

## Auswirkungen in der Praxis

### Verzerrung durch ausgelassene Variablen

Im ersten Kapitel führte das Weglassen des *Geschlechts* aus einem Lohnmodell
dazu, dass das Modell die Löhne von Frauen um etwa 5 $/Std. überschätzte und die
von Männern unterschätzte. Derselbe Mechanismus tritt immer dann auf, wenn eine
relevante, aber unbequeme Variable ausgeschlossen wird.

Beispiele aus der Praxis:
- **Lohngerechtigkeits-Audits**, die nach Berufsbezeichnung und
  Betriebszugehörigkeit kontrollieren, aber die Tatsache auslassen, dass Frauen
  sich in schlechter bezahlten Berufsfeldern konzentrieren, wodurch eine
  strukturelle Lücke verschleiert wird.
- **Kreditscoring-Modelle**, die die ethnische Zugehörigkeit ausschließen
  (korrekt, wie gesetzlich vorgeschrieben), aber Postleitzahl oder Bildung
  einbeziehen, die stark mit ethnischer Zugehörigkeit korrelieren und die
  Verzerrung durch eine Hintertür wieder einführen.
- **Medizinische Risiko-Scores**, die anhand der Behandlungskosten statt der
  Krankheitsschwere trainiert wurden; da Schwarze Patientinnen und Patienten
  historisch weniger Behandlung erhielten, unterschätzte das Modell systematisch
  ihren medizinischen Bedarf.

### Stichprobenverzerrung

Im zweiten Kapitel bedeutete das Training nur mit jungen Arbeitnehmern, dass das
Modell nie lernte, dass Löhne mit der Erfahrung stark ansteigen. Angewendet auf
eine 50-jährige Person sagte es einen Lohn wie für junge Arbeitnehmer voraus.

Beispiele aus der Praxis:
- **Gesichtserkennungssysteme**, die überwiegend mit hellhäutigen Gesichtern
  trainiert wurden und bei dunkleren Hauttönen schlecht abschneiden — teilweise
  mit zehnmal höheren Falsch-Positiv-Raten.
- **Ergebnisse klinischer Studien**, die überwiegend von männlichen oder
  westlichen Populationen stammen und dann auf Gruppen angewendet werden, die in
  der ursprünglichen Studie nicht vertreten waren.
- **Empfehlungssysteme**, die auf Daten von Power-Usern beruhen (die viele
  Artikel bewerten) und gelegentlichen Nutzerinnen und Nutzern schlecht dienen.

### Survivorship Bias

Im dritten Kapitel fehlten in einem Personalarchiv genau die Arbeitnehmer mit
kurzer Betriebszugehörigkeit und niedrigem Lohn, die das Unternehmen bereits
verlassen hatten — genau die Gruppe, die am schwersten vorherzusagen ist. Ein auf
dem Archiv trainiertes Modell schnitt bei langjährigen Mitarbeitenden gut ab,
aber schlecht bei Neueinstellungen.

Beispiele aus der Praxis:
- **Gründungsforschung**, die sich auf aktuell tätige Unternehmen stützt,
  ignoriert die Mehrheit der gescheiterten Start-ups und führt zu übertrieben
  optimistischen Schlüssen darüber, was Unternehmen erfolgreich macht.
- **Anlagestrategien**, die aus heute noch existierenden Fonds abgeleitet werden
  und die vielen Fonds ignorieren, die nach schlechter Performance geschlossen
  wurden — ein bekanntes Problem beim Backtesting.
- **Historische Gehaltsvergleiche**, die auf Beschäftigten beruhen, die im
  Unternehmen geblieben sind, und jene ausschließen, die für besseres Gehalt
  woanders hingewechselt sind, wodurch die tatsächliche Marktrate unterschätzt
  wird.

### P-Hacking

Im dritten Kapitel führten t-Tests auf 20 vollkommen zufälligen Ja/Nein-Merkmalen
allein durch Zufall zu ein oder zwei "signifikanten" Ergebnissen. Würden nur
diese berichtet, enthielte die Literatur eine falsche Aussage über Löhne.

Beispiele aus der Praxis:
- **Medikamentenstudien**, die viele Dosierungen, Subpopulationen oder Endpunkte
  testen und nur diejenigen berichten, die die Signifikanzschwelle überschritten
  haben, was zu einer Replikationskrise in Medizin und Psychologie beiträgt.
- **Wirtschaftspolitische Analysen**, bei denen Forschende viele
  Modellspezifikationen testen (unterschiedliche Kontrollvariablen,
  Stichprobeneinschränkungen, Zeitfenster) und die Spezifikation berichten, die
  eine bevorzugte Schlussfolgerung am besten unterstützt.
- **A/B-Tests** in der Produktentwicklung, bei denen viele gleichzeitige Tests
  laufen und man auf den ersten reagiert, der *p* < 0,05 erreicht, was die
  Falsch-Positiv-Rate weit über die nominellen 5 % hinaus erhöht.

---

## Wie man jede Verzerrung erkennt und abmildert

### Erkennung

| Verzerrung | Wichtige Diagnose |
|---|---|
| **Ausgelassene Variable** | Prüfe, ob die Residuen über eine Gruppe hinweg systematisch sind, die *nicht* als Variable einbezogen wurde |
| **Stichprobe** | Vergleiche die Verteilung wichtiger Variablen in den Trainingsdaten mit der Zielpopulation; bewerte RMSE oder Genauigkeit nach Untergruppe |
| **Survivorship** | Frage: *Wer ist nicht in diesem Datensatz, und warum?* Achte auf unplausibel niedrige Fehlschlags- oder Abbruchraten |
| **P-Hacking** | Prüfe, ob alle Tests und Modellspezifikationen berichtet wurden; stelle die vollständige Verteilung der p-Werte dar. Sie sollte unter der Nullhypothese annähernd gleichverteilt sein |

### Abmilderung

| Verzerrung | Praktische Abhilfe |
|---|---|
| **Ausgelassene Variable** | Alle theoretisch relevanten Variablen einbeziehen; Residuenplots nutzen, um verbleibende Muster auf Gruppenebene zu prüfen |
| **Stichprobe** | Repräsentative Daten erheben; falls unmöglich, die Lücke dokumentieren und Modelle an zurückgehaltenen Daten unterrepräsentierter Gruppen bewerten |
| **Survivorship** | Daten zu Nicht-Überlebenden suchen (geschlossene Fonds, gescheiterte Firmen, Abbrecher); überlebende Fälle explizit gewichten oder kennzeichnen |
| **P-Hacking** | Hypothesen vor der Datenerhebung vorab registrieren; Korrekturen für multiples Testen anwenden (z. B. Bonferroni); alle Tests berichten, nicht nur die signifikanten |

---

## Ethische Überlegungen

Die in diesem Tutorial besprochenen Verzerrungen sind keine abstrakten
statistischen Kuriositäten. Sie betreffen reale Menschen. Ein Lohnmodell, das das
Geschlecht auslässt, kann zur Gehaltsfestlegung verwendet werden. Ein
Kreditmodell, das mit nicht repräsentativen Daten trainiert wurde, kann
qualifizierten Antragstellerinnen und Antragstellern Kredite verweigern. Eine
veröffentlichte, p-gehackte Studie kann in die Politikgestaltung einfließen.

Ein paar Prinzipien, die man im Hinterkopf behalten sollte:

- **Transparenz**: Dokumentiere klar, welche Variablen das Modell verwendet, mit
  welchen Daten es trainiert wurde und welche Gruppen unterrepräsentiert sind.
  Nutzerinnen und Nutzer eines Modells können Verzerrungen nicht korrigieren, von
  denen sie nichts wissen.
- **Verantwortlichkeit**: Lege fest, wer dafür verantwortlich ist, die
  Modellleistung über die Zeit zu überwachen. Verzerrungen können entstehen oder
  sich verschieben, wenn sich die Welt verändert, selbst wenn das Modell selbst
  gleich bleibt.
- **Nach Gruppe bewerten, nicht nur insgesamt**: Die Gesamtgenauigkeit oder der
  Gesamt-RMSE können akzeptabel aussehen, während sie schwere Ungleichheiten für
  bestimmte Untergruppen verschleiern. Schlüssle Leistungsmetriken immer nach den
  Gruppen auf, die dein Modell betreffen wird.
- **Alle Analysen berichten**: Ob in der Wissenschaft oder in der Industrie — nur
  die Analyse zu berichten, die das attraktivste Ergebnis liefert, ist eine Form
  von P-Hacking. Ein transparenter Arbeitsablauf dokumentiert, was versucht und
  was gefunden wurde, einschließlich Nullresultaten.

---

## Übungen

### Übung 1: Verzerrung durch ausgelassene Variablen
```{raw} html
<style>
  .quiz-container {
    background: #fff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 2rem;
    max-width: 750px;
    font-family: "Segoe UI", Roboto, sans-serif;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
  }
  .quiz-container p.question {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1.5rem;
  }
  .quiz-container label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.75rem 0;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background-color: #eef2ff;
    cursor: pointer;
  }
  .quiz-container button {
    margin-top: 1.5rem;
    background-color: #4f46e5;
    color: white;
    padding: 0.6rem 1.2rem;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
  }
  .quiz-feedback {
    margin-top: 1.25rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-weight: 500;
  }
  .success { background-color: #dcfce7; color: #16a34a; border: 1px solid #16a34a; }
  .error { background-color: #fee2e2; color: #dc2626; border: 1px solid #dc2626; }
  .warning { background-color: #fef9c3; color: #a16207; border: 1px solid #facc15; }
</style>

<div class="quiz-container" id="quiz1">
  <p class="question">In Kapitel 1 haben wir ein Lohnmodell erstellt, das die Variable <em>Geschlecht</em> ausgelassen hat. Was war die Hauptfolge?</p>
  <form id="quiz1-form">
    <label><input type="radio" name="q1" value="a"> Der Gesamt-RMSE des Modells wurde null</label>
    <label><input type="radio" name="q1" value="b"> Das Modell überschätzte systematisch die Löhne von Frauen und unterschätzte die von Männern</label>
    <label><input type="radio" name="q1" value="c"> Das Modell verweigerte jede Vorhersage</label>
    <label><input type="radio" name="q1" value="d"> Das Modell wurde genauer, weil es weniger Eingaben verarbeiten musste</label>
    <button type="button" onclick="checkQ1()">Antwort abschicken</button>
    <p id="quiz1-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ1() {
  const answers = {
    a: { correct: false, feedback: "Falsch. Das Auslassen einer Variable erhöht den Fehler, es beseitigt ihn nicht." },
    b: { correct: true,  feedback: "Richtig! Ohne das Geschlecht im Modell wendete es auf alle eine einzige Durchschnittsformel an. Da Männer in den Daten im Schnitt mehr verdienen, überschätzte das Modell die Löhne von Frauen und unterschätzte die von Männern um jeweils etwa 5 $/Std." },
    c: { correct: false, feedback: "Falsch. Das Modell lief weiterhin, aber seine Vorhersagen waren für jede Gruppe systematisch verzerrt." },
    d: { correct: false, feedback: "Falsch. Das Entfernen einer relevanten Variable erhöht die Verzerrung. Es hilft nicht." }
  };
  const selected = document.querySelector('input[name="q1"]:checked');
  const feedback = document.getElementById("quiz1-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Bitte wähle eine Antwort aus.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

### Übung 2: Stichprobenverzerrung
```{raw} html
<div class="quiz-container" id="quiz2">
  <p class="question">In Kapitel 2 wurde das verzerrte Modell nur mit jungen Arbeitnehmern trainiert. Warum schnitt es bei älteren Arbeitnehmern zur Testzeit so schlecht ab?</p>
  <form id="quiz2-form">
    <label><input type="radio" name="q2" value="a"> Das Modell hatte zu viele Parameter und überfittete auf die jungen Arbeitnehmer</label>
    <label><input type="radio" name="q2" value="b"> Der Entscheidungsbaum hatte während des Trainings nie Arbeitnehmer mit viel Erfahrung gesehen und konnte deshalb ihre Löhne nicht vorhersagen</label>
    <label><input type="radio" name="q2" value="c"> Die Löhne älterer Arbeitnehmer lassen sich aus den verfügbaren Variablen schlicht nicht vorhersagen</label>
    <label><input type="radio" name="q2" value="d"> Die Testmenge enthielt zu wenige ältere Arbeitnehmer für eine ordentliche Bewertung</label>
    <button type="button" onclick="checkQ2()">Antwort abschicken</button>
    <p id="quiz2-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ2() {
  const answers = {
    a: { correct: false, feedback: "Falsch. Overfitting bedeutet, dass sich das Modell zu eng an das Rauschen der Trainingsdaten anpasst. Das ist ein anderes Problem als die Stichprobenverzerrung." },
    b: { correct: true,  feedback: "Richtig! Ein Entscheidungsbaum lernt Regeln nur aus den Daten, die er gesehen hat. Da in der verzerrten Trainingsmenge keine mittelalten oder älteren Arbeitnehmer vorkamen, erfassten die Blätter des Modells nur Lohnniveaus, wie sie für junge Arbeitnehmer typisch sind (10–16 $/Std.). Jede Testperson mit 20+ Jahren Erfahrung landete in diesen Blättern und erhielt eine Vorhersage wie für junge Arbeitnehmer, weit unter dem tatsächlichen Lohn." },
    c: { correct: false, feedback: "Falsch. Das ausgewogene Modell sagte die Löhne älterer Arbeitnehmer genau vorher. Das Problem lag in den Trainingsdaten, nicht in der Aufgabe." },
    d: { correct: false, feedback: "Falsch. Die Testmenge war für beide Modelle identisch; der Unterschied kam ausschließlich aus den Trainingsdaten." }
  };
  const selected = document.querySelector('input[name="q2"]:checked');
  const feedback = document.getElementById("quiz2-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Bitte wähle eine Antwort aus.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

### Übung 3: Survivorship Bias
```{raw} html
<div class="quiz-container" id="quiz3">
  <p class="question">In Kapitel 3 fehlten im Unternehmensarchiv Arbeitnehmer mit kurzer Betriebszugehörigkeit und unterdurchschnittlichem Lohn. Was beschreibt die Auswirkung auf ein mit diesem Archiv trainiertes Modell am besten?</p>
  <form id="quiz3-form">
    <label><input type="radio" name="q3" value="a"> Das Modell wurde fairer, weil Geringverdiener ausgeschlossen wurden</label>
    <label><input type="radio" name="q3" value="b"> Das Modell schnitt bei Arbeitnehmern mit langer Zugehörigkeit gut ab, aber schlecht bei solchen mit kurzer Zugehörigkeit, deren Lohnmuster es kaum gesehen hatte</label>
    <label><input type="radio" name="q3" value="c"> Das Modell überschätzte die Löhne von Arbeitnehmern mit langer Zugehörigkeit</label>
    <label><input type="radio" name="q3" value="d"> Es gab keine Auswirkung, weil die Betriebszugehörigkeit als Variable einbezogen wurde</label>
    <button type="button" onclick="checkQ3()">Antwort abschicken</button>
    <p id="quiz3-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ3() {
  const answers = {
    a: { correct: false, feedback: "Falsch. Geringverdiener auszuschließen macht das Modell nicht fairer. Es macht es blind für einen wichtigen Teil der Population." },
    b: { correct: true,  feedback: "Richtig! Das Archiv enthielt vor allem Arbeitnehmer, die geblieben waren, weil sie gut verdienten. Arbeitnehmer mit kurzer Zugehörigkeit und niedrigem Lohn hatten das Unternehmen größtenteils verlassen und waren für das Modell unsichtbar. Als das Modell an der wahren Population getestet wurde (einschließlich jener, die gegangen wären), waren seine Fehler für die Gruppe mit kurzer Zugehörigkeit viel größer als für die mit langer Zugehörigkeit." },
    c: { correct: false, feedback: "Falsch. Das Modell hatte reichlich Daten zu Arbeitnehmern mit langer Zugehörigkeit, daher waren seine Vorhersagen für sie relativ genau." },
    d: { correct: false, feedback: "Falsch. Die Betriebszugehörigkeit als Variable einzubeziehen, gleicht nicht aus, dass die meisten Arbeitnehmer mit kurzer Zugehörigkeit und niedrigem Lohn schlicht nicht in den Trainingsdaten vorkamen." }
  };
  const selected = document.querySelector('input[name="q3"]:checked');
  const feedback = document.getElementById("quiz3-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Bitte wähle eine Antwort aus.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

### Übung 4: P-Hacking
```{raw} html
<div class="quiz-container" id="quiz4">
  <p class="question">In Kapitel 3 haben wir 20 vollkommen zufällige Ja/Nein-Merkmale auf einen Lohneffekt getestet und ein oder zwei "signifikante" Ergebnisse (p &lt; 0,05) gefunden. Was ist die richtige Interpretation?</p>
  <form id="quiz4-form">
    <label><input type="radio" name="q4" value="a"> Die signifikanten Merkmale beeinflussen tatsächlich den Lohn und sollten berichtet werden</label>
    <label><input type="radio" name="q4" value="b"> Das Ergebnis bestätigt, dass bestimmte persönliche Eigenschaften das Einkommen tatsächlich beeinflussen</label>
    <label><input type="radio" name="q4" value="c"> Bei 20 unabhängigen Tests mit einer Schwelle von 5 % ist rein durch Zufall etwa ein falsch positives Ergebnis zu erwarten</label>
    <label><input type="radio" name="q4" value="d"> Die p-Wert-Schwelle hätte bei 1 % statt bei 5 % liegen sollen</label>
    <button type="button" onclick="checkQ4()">Antwort abschicken</button>
    <p id="quiz4-feedback" class="quiz-feedback" style="display:none;"></p>
  </form>
</div>

<script>
function checkQ4() {
  const answers = {
    a: { correct: false, feedback: "Falsch. Die Merkmale wurden zufällig erzeugt und haben keinen echten Zusammenhang mit dem Lohn. Ein signifikanter p-Wert bedeutet nur, dass das Ergebnis unter der Nullhypothese unwahrscheinlich ist. Er beweist keinen echten Effekt, besonders wenn viele Tests durchgeführt wurden." },
    b: { correct: false, feedback: "Falsch. Die Merkmale (z. B. 'besitzt ein Haustier', 'bevorzugt Berge') wurden zufällig zugewiesen und können keine echten Lohneffekte widerspiegeln." },
    c: { correct: true,  feedback: "Richtig! Bei 20 Tests und einer Schwelle von 5 % erwarten wir 20 × 0,05 = 1 falsch positives Ergebnis, selbst wenn nichts wirklich signifikant ist. Nur das 'signifikante' Ergebnis zu berichten, ohne die anderen 19 Tests zu erwähnen, ist P-Hacking. Die übliche Abhilfe ist, alle Tests zu berichten und bei vielen Vergleichen eine Korrektur wie die Bonferroni-Anpassung anzuwenden." },
    d: { correct: false, feedback: "Teilweise richtig, dass eine strengere Schwelle falsch positive Ergebnisse reduziert, aber das allein löst nicht das Problem, viele Tests durchzuführen und nur die signifikanten zu berichten. Der Kernpunkt ist Transparenz über alle durchgeführten Tests." }
  };
  const selected = document.querySelector('input[name="q4"]:checked');
  const feedback = document.getElementById("quiz4-feedback");
  feedback.style.display = "block";
  if (!selected) {
    feedback.textContent = "Bitte wähle eine Antwort aus.";
    feedback.className = "quiz-feedback warning";
    return;
  }
  const result = answers[selected.value];
  feedback.textContent = result.feedback;
  feedback.className = "quiz-feedback " + (result.correct ? "success" : "error");
}
</script>
```

---

## Zusammenfassung

In den drei vorangegangenen Kapiteln sind wir vier verschiedenen Wegen begegnet,
auf denen ein Modell systematisch falsche Ergebnisse liefern kann:

1. **Verzerrung durch ausgelassene Variablen** entsteht, wenn eine Variable, die
   das Ergebnis tatsächlich beeinflusst, im Modell fehlt. Das Modell kann
   Gruppenunterschiede, die es nicht sehen kann, nicht berücksichtigen, und seine
   Fehler werden systematisch statt zufällig.

2. **Stichprobenverzerrung** entsteht, wenn die Trainingsdaten nicht die
   Population widerspiegeln, auf die das Modell angewendet wird. Gruppen, die im
   Training selten vorkommen, werden schlecht vorhergesagt, und die
   Gesamtgenauigkeit kann das vollständig verschleiern.

3. **Survivorship Bias** ist ein Spezialfall der Stichprobenverzerrung: Nur die
   "Überlebenden" eines Auswahlprozesses erscheinen in den Daten. Das Modell
   lernt etwas über erfolgreiche oder stabile Fälle und kann nicht auf jene
   verallgemeinern, die nicht überlebt haben.

4. **P-Hacking** ist keine Verzerrung in den Daten, sondern in der Analyse: Viele
   Tests durchzuführen und nur die signifikanten zu berichten, erhöht die
   Falsch-Positiv-Rate und führt zu Befunden, die sich nicht replizieren lassen.

Der gemeinsame Faden ist, dass **das, was in den Daten oder im Bericht fehlt,
genauso wichtig ist wie das, was vorhanden ist**. Die Frage "Wer oder was fehlt
hier, und warum?" zu stellen, ist eine der mächtigsten Fragen, die sich eine
Forschungsperson oder Praktikerin oder Praktiker stellen kann, bevor sie der
Ausgabe eines Modells vertraut.
