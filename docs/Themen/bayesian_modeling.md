# Bayesian Modeling
von Serife-Nur Özdemir, Sanamjeet Meyer und Anna Postnikova

Abstract beschreibt kurz und in wenigen Sätzen ihr Thema. Der Abstract dient dem Lesenden als Orientierungshilfe ob er/sie weiterlesen möchten.

Die Bayes'sche Modellierung ist ein Konzept, das auf der Bayes'schen Statistik basiert. Der Bayes'sche Ansatz spiegelt im Vergleich zum Frequentistischen Ansatz eher die menschliche Denkweise wider. Um genauer zu sein, berücksichtigt der Bayes'sche Ansatz das Vorwissen bei der Inferenz. Dies ist besonders hilfreich, wenn bereits Expertenwissen oder Vorkenntnisse in einem Bereich vorhanden sind. Fast alle klassischen Modelle des maschinellen Lernens können in Bayessche Modelle umgewandelt werden. Was Bayesian Modeling ist, wie es genau funktioniert, welche Vor- und Nachteile es hat, darauf gehen wir in dieser Arbeit ein.

Die folgenden Gliederung ist Beispielhaft und kann von Ihnen nach Wunsch angepasst werden. Am Ende sollte ein ca. 10-15 Seiten langes Dokument vorliegen. Falls Sie weitere Infos zur Formatierung benötigen schauen Sie in der [Referenz](https://squidfunk.github.io/mkdocs-material/reference/).
## Vergeleich Frequentist und Bayes
- Vorteile von Bayesian Modelling
## Einführung in die Bayes Statistik

### Bayes-Theorem

Das Bayes-Theorem ist ein grundlegendes Konzept der Wahrscheinlichkeitstheorie, das es uns ermöglicht, unsere Überzeugungen über ein Ereignis basierend auf neuen Beweisen zu aktualisieren. Es kann wie folgt formuliert werden:


Wo:
- P(A|B) repräsentiert die Wahrscheinlichkeit, dass Ereignis A eintritt, unter der Bedingung, dass Ereignis B eingetreten ist.
- P(B|A) ist die Wahrscheinlichkeit, dass Ereignis B eintritt, unter der Bedingung, dass Ereignis A eingetreten ist.
- P(A) ist die Wahrscheinlichkeit, dass Ereignis A eintritt.
- P(B) bezeichnet die Wahrscheinlichkeit, dass Ereignis B eintritt.

$$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$

Das Bayes-Theorem bietet eine Möglichkeit, die Wahrscheinlichkeit von Ereignis A unter Berücksichtigung vorhandener Kenntnisse und neuer Beweise zu berechnen. Es wird in verschiedenen Bereichen eingesetzt, darunter Statistik, maschinelles Lernen und Datenanalyse.

### Anwendungsbeispiel Medizinischer Test

- In 99,5% der Fälle fällt der Test positiv aus.
- Sollte die Krankheit nicht vorliegen, beträgt die Wahrscheinlichkeit für einen positiven Test 1%.
- Laut einer Studie leidet eine von vier Personen an der betreffenden Krankheit.
- Wie groß ist die Wahrscheinlichkeit, dass jemand an der Krankheit leidet, obwohl der Test ein negatives Ergebnis zeigt?

### Modell-Annahmen

K = Person ist krank  
T = Test fällt positiv aus

\[
P(T|K) = 0.995
\]

\[
P(T|\overline{K}) = 0.01
\]

\[
P(K) = 0.25
\]

### Gesucht wird die Wahrscheinlichkeit

\[
P(K|\overline{T})
\`

## Anwendungsbeispiel Bayes Theorem

### Bayes Theorem

\[
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
\`

\[
P(K|\overline{T}) = \frac{P(\overline{T}|K)P(K)}{P(\overline{T}|K)P(K) + P(\overline{T}|\overline{K})P(\overline{K})}
\`

\[
= \frac{(1-0.995)\cdot 0.25}{(1-0.995)\cdot 0.25 + (1-0.01)\cdot 0.75}
\`

\[
= 0.00185
\`







- Bayes Theorem
- Inferenz
## Methoden
- MCMC
## Anwendungen
- Bayesian Logoistic Regression
- Bayes Linear Regression
- Bayes NN
- Bayesian Optimization
## Fazit
- Vor und Nachteile von Bayesian Modelling
## Weiterführendes Material
- Relevante links und online Kurse
### Podcast
Hier Link zum Podcast.

### Talk
Hier einfach Youtube oder THD System embedden.

### Demo
Hier Link zum Demo Video + Link zum GIT Repository mit dem Demo Code.


## Literaturliste
Hier können Sie auf weiterführende Literatur verlinken. 
