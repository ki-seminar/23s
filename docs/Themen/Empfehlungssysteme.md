# Empfehlungssysteme
von *Olha Solodovnyk, Daria Likhacheva und Zi Xun Tan*

## Abstract 

## 1 Einleitung / Motivation
In der heutigen digitalen Welt werden die Empfehlungssysteme fast jeden Tag benutzt. Sie haben einen sehr großen Einfluss auf unsere täglichen Aktivitäten. Von Online-Shopping-Plattformen bis hin zu Streaming-Dienste (wie Spotify oder Netflix) werden die Empfehlungen benutzt, um unsere Entscheidungen zu beeinflussen und unsere Erfahrungen zu personalisieren. Was bedeuten aber diese Empfehlungssysteme und wie funktionieren sie?

Wir werden immer mit relevanten Informationen, Produkten und Dienstleistungen versorgt, da die Empfehlungssysteme dafür verantwortlich sind. Für die Empfehlungssysteme zu funktionieren, brauchen wir sehr viele Daten, die aus unseren Vorlieben, Nutzerverhalten und Aktivitäten im Netz gesammelt und analysiert werden. Mit ständigen Empfehlungen von neuen Filmen, Bücher, Musik, Produkte und Dienstleistungen haben die Empfehlungssysteme das Potenzial, unser Leben immer mehr einfacher und bequemer zu gestalten. Sie verbessern aber nicht nur das Leben von Nutzern, sondern auch den Unternehmen und Künstlern. Indem sie ihnen eine Möglichkeit bieten, ihre Werke einem breiteren Publikum zu präsentieren oder über ihre Dienstleistungen zu informieren, können sie dazu beitragen, die Sichtbarkeit von Künstlern, Autoren, Filmemachern, Dienstleistungsanbieter und Unternehmen zu erhöhen.

Dennoch begegnen uns in der modernen Welt auch spezifische Herausforderungen im Zusammenhang mit Empfehlungssystemen. Unter den ethischen Fragen, die berücksichtigt werden müssen, sind die Privatsphäre von Kunden, Manipulation und Voreingenommenheit der Empfehlungsalgorithmen. 

Im Folgenden wollen wir einen Blick auf die Welt der Empfehlungssysteme werfen, die verschiedenen Arten der Empfehlungssysteme uns anschauen, ihre Funktionsweise verstehen, ihre Vor- und Nachteile beleuchten und die Auswirkungen auf unsere Gesellschaft sowie individuellen Entscheidungsprozesse untersuchen. Wenn wir und das Thema genauer anschauen, können wir besser verstehen, wie Empfehlungssysteme unseren Alltag verbessern und wie wir als Nutzerinnen und Nutzer bewusste Entscheidungen treffen können.
## 2 Methoden

### 2.1 Inhaltsbasierte Filterung

### 2.2 Kollaborative Filterung
Die kollaborative Filterung ist ein Ansatz, der die vergangenen Bewertungen eines Benutzers nutzt, um eine Datenbank (Benutzer-Objekt-Matrix) von Präferenzen zu erstellen und Vorhersagen über Gegenstände zu treffen, die mit den Vorlieben des Benutzers übereinstimmen. Sie wird in zwei Hauptkategorien unterteilt: memory-basierte kollaborative Filterung und modellbasierte kollaborative Filterung.

#### 2.2.1 Memory-basierte kollaborative Filterung

#### 2.2.2 Modellbasierte kollaborative Filterung
Bei der Verwendung von memory-basierte Methoden entsteht das Problem, dass die Benutzer-Objekt-Matrix aufgrud einer großen Anzahl von fehlenden Bewertungen sehr dünn besetzt ist. Diese Datenspärlichkeit beeinträchtigt die Fähigkeit, Benutzervorlieben genau einzuschätzen und zuverlässige Empfehlungen zu geben. 

Ein Lösungsansatz für dieses Problem besteht in der Verwendung von modellbasierten Methoden. Im Gegensatz zu memory-basierte Methoden lernen modellbasierte Algorithmen aus den vorhandenen Bewertungen ein Modell, das dann zur Vorhersage fehlender Bewertungen genutzt werden kann. Diese Modelle können auf Techniken wie Clustering, Bayes'schen Klassifikatoren oder Matrixfaktorisierung basieren.

**Singular Value Decomposition (SVD) / Singulärwertzerlegung**  
Obwohl es verschiedene Algorithmen gibt, konzentrieren wir uns hauptsächlich auf die Matrixfaktorisierung mit SVD. Die zentrale Idee bei der SVD-basierten Matrixfaktorisierung besteht darin, eine niedrigdimensionale Approximation der ursprünglichen Bewertungsmatrix zu finden. Diese niedrigdimensionale Approximation ermöglicht es uns, die wichtigsten latenten Faktoren zu erfassen, die zu Benutzervorlieben und Objekteigenschaften beitragen. In einem Filmempfehlungssystem könnten latenten Faktoren beispielsweise zugrunde liegende Attribute wie Genre, Regisseur oder Schauspieler darstellen.

Angenommen, wir haben eine Matrix $A_{m \times p}$, in der die $m$ Zeilen die Benutzer repräsentieren und die $p$ Spalten die Objekte darstellen. Das Ziel des SVD-Theorems besteht darin, die hochdimensionale Matrix $A_{m \times p}$ in drei Matrizen mit geringerer Dimension aufzuteilen. Das Theorem besagt:

$$
A = UΣV^T
$$

$U$ steht für die Benutzermatrix, $Σ$ ist eine diagonale Matrix mit den Singulärwerten und $V^T$  bezeichnet die Objektmatrix. Die Singulärwerte in der $Σ$-Matrix werden in absteigender Reihenfolge sortiert. Durch Auswahl der obersten $k$ Singulärwerte und ihrer entsprechenden Spalten in $U$ und $V^T$ reduzieren wir die Dimensionalität der Matrizen. Der Wert von $k$ bestimmt die Anzahl der beibehaltenen latenten Faktoren. 

Wenn wir uns das Filmempfehlungssystem als Beispiel nehmen, geht dieser Ansatz davon aus, dass es versteckte Beziehungen zwischen Benutzern und Filmen gibt, die sich auf die Bewertung eines Benutzers für einen bestimmten Film auswirken. Konkret wird angenommen, dass es eine Reihe von $k$ Faktoren gibt, die bestimmen, wie ein Benutzer einen Film bewertet, und dass diese Faktoren durch die Rang-$k$ SVD erfasst werden können.

**Training des Modells mit Surprise Library**  
In der Surprise-Bibliothek ist SVD als Empfehlungssystem-Modul implementiert. Im Wesentlichen geht es darum, das Empfehlungsproblem in ein Optimierungsproblem umzuwandeln. Wir können die Güte unserer Vorhersagen für die Nutzerbewertungen von Objekten messen. Eine häufig verwendete Metrik dafür ist der Root Mean Squared Error (RMSE). Je niedriger der RMSE, desto besser die Güte.

```python
from surprise import SVD

# Create a reader object
reader = Reader()

# Load the dataset from the DataFrame, specifying the columns for userId, movieId, and rating
data = Dataset.load_from_df(df_ratings[["userId", "movieId", "rating"]], reader)

# Instantiate the SVD algorithm
svd = SVD()

# Run 5-fold cross-validation and print results
cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=True)
```
Um eine Bewertung für einen bestimmten Benutzer vorherzusagen, kann die Methode `predict()` verwendet werden. Beachten Sie jedoch, dass der Benutzer in Ihrer Datenbank vorhanden sein muss.
```python
predicted_rating = svd.predict(user_id=1, movie_id=12)
predicted_rating.est
```
**Hyperparameter Tuning**  
Bei der Optimierung des SVD-Modells in der Surprise-Bibliothek gibt es mehrere wichtige Hyperparameter zu beachten:

`n_factors`: Dieser Hyperparameter bestimmt die Anzahl der latenten Faktoren oder Dimensionen, die zur Darstellung von Benutzern und Objekten verwendet werden. Durch Erhöhung der Anzahl der Faktoren können möglicherweise komplexere Beziehungen erfasst werden, es kann jedoch auch zu Overfitting kommen.

`n_epochs`: Dieser Hyperparameter repräsentiert die Anzahl der Iterationen oder Epochen, die während des Optimierungsprozesses verwendet werden. Durch Erhöhung der Anzahl der Epochen kann das Modell besser aus den Daten lernen, aber zu viele Epochen können ebenfalls zu Overfitting führen.

`lr_all`: Dieser Hyperparameter steuert die Lernrate, die den Schrittweite im Optimierungsalgorithmus bestimmt. Eine höhere Lernrate kann zu schnellerer Konvergenz führen, aber auch dazu führen, dass das optimale Ergebnis überschritten wird.

`reg_all`: Dieser Hyperparameter steuert den Regularisierungsterm für alle Parameter im Modell. Regularisierung hilft, Overfitting zu verhindern, indem eine Strafe für komplexe Modelle hinzugefügt wird. Die Anpassung dieses Hyperparameters kann einen Kompromiss zwischen der Erfassung nützlicher Muster und der Vermeidung von Overfitting darstellen.
### 2.3 Hybride Empfehlungssyteme 
Sowohl das Inhaltsbasierte als auch das kollaborative Filtermodell haben ihre Einschränkungen. Inhaltsbasierte Empfehlungssysteme haben die Einschränkung, dass sie stark von den verfügbaren Merkmalen oder Attributen eines Objekts abhängen, um Ähnlichkeiten zu ermitteln. Dadurch können sie Schwierigkeiten haben, komplexe und subtile Zusammenhänge zwischen verschiedenen Objekten zu erfassen, die sich nicht einfach durch Attribute beschreiben lassen.

Auf der anderen Seite können kollaborative Filterungssysteme Einschränkungen aufweisen, wenn es um die Bewältigung des sogenannten "Cold-Start"-Problems geht. Dieses Problem tritt auf, wenn ein neuer Benutzer oder ein neues Objekt in das System eingeführt wird und keine ausreichenden Informationen über die Vorlieben oder Ähnlichkeiten zu anderen Benutzern oder Objekten vorliegen. Dadurch kann die Genauigkeit der Empfehlungen in solchen Situationen beeinträchtigt werden.

Die Idee hinter hybriden Techniken besteht darin, dass eine Kombination von Algorithmen genauere und effektivere Empfehlungen liefert als ein einzelner Algorithmus, da die Nachteile eines Algorithmus durch einen anderen Algorithmus überwunden werden können. Indem verschiedene Empfehlungsmethoden miteinander verbunden werden, können hybride Filtertechniken die Stärken der einzelnen Ansätze nutzen und gleichzeitig deren Schwächen reduzieren. Dadurch ermöglichen sie eine verbesserte Personalisierung und Präzision bei den Empfehlungen für Benutzer.

Das Hybrid-Empfehlungsmodell ist in sieben Typen unterteilt: 

| Typ | Beschreibung | Vorteile | Beispiel |
|-----|--------------|----------|----------|
| Gewichtet | Eine Methode, bei der die Gewichtung allmählich angepasst wird, je nachdem, inwieweit die Bewertung eines Gegenstandes durch den Benutzer mit der durch das Empfehlungssystem vorhergesagten Bewertung übereinstimmt. | Nutzung der Stärken verschiedener Empfehlungssysteme, einfache Implementierung. | P-tango |
| Schalten | Wechselt zwischen Empfehlungstechniken basierend auf einer heuristischen Methode, die die Fähigkeit zur Erzeugung guter Bewertungen berücksichtigt. Löst spezifische Probleme einzelner Methoden. | Vermeidet Probleme spezifischer Methoden, sensibel für Stärken und Schwächen der Empfehlungssysteme. | DailyLearner |
| Kaskade   |Nach der Erstellung einer Kandidatenliste unter Verwendung eines Empfehlungssystemmodells mit ähnlichem Geschmack wie der Benutzer kombiniert die Methode das zuvor verwendete Empfehlungssystemmodell mit einem anderen Modell, um die Kandidatenliste nach den für den Benutzer am besten geeigneten Artikeln zu sortieren. | Verfeinert Empfehlungen durch Iteration, effizient und tolerant gegenüber Störungen. | EntreeC |
| Gemischt  | Kombiniert Empfehlungsergebnisse verschiedener Techniken gleichzeitig für jeden Artikel und liefert mehrere Empfehlungen. | Bietet mehrere Empfehlungen pro Artikel, individuelle Leistungen beeinflussen die Gesamtleistung in einem begrenzten Bereich nicht. | PTV-System, Profinder, PickAFlick |
| Merkmalskombination | Bezieht sich auf die Integration von Merkmalen, die von einer Empfehlungstechnik erzeugt wurden, in eine andere Technik. Durch die Einbeziehung spezifischer Merkmale oder Bewertungen einer Technik als zusätzliche Eingabe für eine andere Technik wird der Empfehlungsprozess verbessert, indem verschiedene Informationsquellen genutzt werden, um die Genauigkeit und Relevanz der Empfehlungen zu steigern. | Nutzt kollaborative Daten in Verbindung mit anderen Techniken, reduziert die Abhängigkeit von der kollaborativen Filterung. | Pipper |
| Funktionserweiterung | Nutzt Bewertungen und zusätzliche Informationen, die von vorherigen Empfehlungssystemen generiert werden. Erfordert zusätzliche Funktionalität der Empfehlungssysteme.  | Fügt dem primären Empfehlungssystem eine geringe Anzahl von Merkmalen hinzu, verbessert die Genauigkeit der Empfehlungen. | Libra-System |
| Meta-Ebene | Verwendet das interne Modell einer Technik als Eingabe für eine andere. Bietet umfassendere Informationen im Vergleich zu einzelnen Bewertungen. | Löst das Problem der Datenspärlichkeit bei kollaborativen Filterungstechniken, nutzt umfassende Modelle für verbesserte Empfehlungen. | LaboUr |


## 3 Anwendungen
Empfehlungssysteme finden in sehr vielen Bereichen Anwendung und tragen dazu bei, die Entscheidungsfindung von Benutzern zu erleichtern. Hier werden einige der Hauptanwendungen von Empfehlungssystemen aufgezeigt:

- **E-Commerce:** Online-Shopping-Plattformen wie Amazon oder Zalando nutzen Empfehlungssysteme, um ihren Kunden Produkte vorzuschlagen, die ihren individuellen Vorlieben entsprechen. Basierend auf dem bisherigen Kaufverhalten, den Produktbewertungen und den Präferenzen anderen Nutzern werden personalisierte Produktvorschläge gemacht, um die Einkaufserfahrung zu verbessern und die Kundenbindung zu stärken.
- **Streaming-Dienste:** Plattformen wie Netflix und YouTube verwenden Empfehlungssysteme, um Nutzern Inhalte vorzuschlagen, die ihren Geschmack treffen. Basierend auf dem Seh- und Hörverhalten sowie den Bewertungen vergangener Inhalte werden individuelle Serienempfehlungen und Musikvorschläge generiert, um das Unterhaltungserlebnis zu personalisieren.
- **Soziale Medien:** Hier setzen Facebook, Instagram und Twitter Empfehlungssysteme ein, um Nutzern relevante Inhalte, Beiträge und Kontakte anzuzeigen. Basierend auf dem sozialen Netzwerk, den Interaktionen und den Vorlieben der Nutzer werden Beiträge von Freunden, Seiten oder Themen empfohlen, um das Engagement und die Nutzung der Plattform zu fördern.
- **Musikdienste:** Musik-Streaming-Dienste wie Spotify oder Apple Musik nutzen Empfehlungssysteme, um personalisierte Wiedergabelisten zu erstellen. "Discover Weekly" und "Release Radar" sind Beispiele dafür, wie Empfehlungssysteme basierend auf dem individuellen Musikgeschmack der Nutzer neue Künstler, Songs und Alben vorschlagen.
- **Nachrichten und Content-Aggregatoren:** Empfehlungssysteme werden auch in Nachrichten-Apps und Content-Aggregatoren eingesetzt, um Nutzern relevante Artikel, Blogposts und Nachrichten vorzuschlagen. Basierend auf den Präferenzen, dem Leseverhalten und den Interessen werden personalisierte Newsfeeds und Inhaltszusammenstellungen erstellt.
- **Reise- und Hotelbuchungen:** Empfehlungssysteme kommen auch in der Tourismusbranche zum Einsatz. Plattformen wie Booking.com oder Airbnb nutzen sie, um Nutzern maßgeschneiderte Hotelvorschläge, Reiserouten und Attraktionen zu präsentieren, die ihren individuellen Vorlieben und Bedürfnissen entsprechen.

Diese Anwendungen von Empfehlungssystemen sind nur einige Beispiele für die vielfältigen Einsatzmöglichkeiten dieser Technologie. Ob im Bereich des Online-Shoppings, der Unterhaltung, der sozialen Interaktion oder der Informationsbeschaffung, Empfehlungssysteme spielen eine zentrale Rolle dabei, unsere Erlebnisse zu verbessern, relevante Inhalte zu entdecken und unsere Zeit effizienter zu nutzen.

## 4 Fazit

## 5 Weiterführendes Material

### 5.1 Podcast
Hier Link zum Podcast.

### 5.2 Talk
Hier einfach Youtube oder THD System embedden.

### 5.3 Demo
Hier Link zum Demo Video  
[Link zum GitHub Repository](https://github.com/Gary0417/movie_recommendation_system)



## 6 Literaturliste
[Ko, H. Y., Lee, S. Y., Park, Y., & Choi, A. L. (2022). A Survey of Recommendation Systems: Recommendation Models, Techniques, and Application Fields. Electronics; MDPI. https://doi.org/10.3390/electronics11010141](https://doi.org/10.3390/electronics11010141)

[Isinkaye, F. O., Folajimi, Y., & Ojokoh, B. A. (2015). Recommendation systems: Principles, methods and evaluation. Egyptian Informatics Journal; Elsevier BV. https://doi.org/10.1016/j.eij.2015.06.005](https://doi.org/10.1016/j.eij.2015.06.005)

