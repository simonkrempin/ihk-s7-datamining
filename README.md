# Aufgaben Teil 1

## Aufgabe A

Die Daten folgen einer ungleichen Verteilung. Insgesamt liegen von 30000 Datenpunkten nur 1746 Betrugsfälle vor, was lediglich **5,82%** ergeben und somit ein stark unausgeglichener Datensatz vorliegt. Relevant ist dabei, dass während der Messung die Imbalancen passend addressiert werden. Im Mittelpunkt dessen steht ein adequates Messinstrument wie der Macro F1-Score, der einen Fokus auf die Klassengewichtung setzt.

## Aufgabe B

Die Spalten ANUMMER\_01 bis ANUMMMER\_10 repräsentieren die Artikelnummern der Bestellungen. Die Repräsentation der Artikel über eine Liste ist dahingehend ungeeignet, da zwei kritische Probleme damit einhergehen. Erstens sind die Bestellungen damit auf 10 begrenzt und Warenkörbe ab 11 Elementen können nicht akurat dargestellt werden. Dazu kommt, dass eine hohe Anzahl an NULL-Werten gespeichert werden muss. Ein Vorteil einer solchen Darstellung ist eine verbesserte Performance der Query, da kein Leistung auf die JOIN-Operation aufgewandt wird. Auf der anderen Seite ermöglicht eine abgewandte Darstellung, indem die Arikel in eine weitere 1-n Tabelle ausgelagert werden, eine flexiblere Handhabung der Artikel. Durch eine Erweiterung zu einer n-m Tabelle, lassen sich ebenfalls Metadaten im normalisierten Zustand hinzufügen, die relevant für die Verbesserung der Vorhersage sein könnten.

## Aufgabe C

Theoretische Grundlagen:

- **Overfitting**: Lernt die Trainingsdaten auswendig
- **Underfitting**: Model ist nicht komplex genug

Für die Implementation wurde das Python Framework Scikit-Learn verwendet. Scikit-Learn bietet eine umfassende Sammlung an Werkzeugen für *predictive data analysis*, darunter Klassifizierung, Regression und Clustering. Für die Aufgaben wurden die Algorithmen Random Forest, Gradient Boosting und Support Vector Machine ausgewählt. Random Forest ist ein Ensemble von Entscheidungsbäumen, die zusammen die Vorhersage ermitteln, entweder über den Durchschnittswert oder Mehrheitsentscheidung. Ähnlich dazu ist Gradient Boosting ein sequenzielles Ensemble, also nacheinander trainierte Entscheidungsbäume, die versuchen die Fehler des vorherigen Baums auszugleichen. Mit den beiden Algorithmen stehen Varianz und Bias gegenüber, oder auch reduziertes Overfitting gegen erhöhte Präzision. Zuguterletzt versucht Support Vector Machine eine Trennlinie zwischen die Datenpunkte zu ziehen, um die Daten zu kategorisieren und basierend darauf eine Vorhersage zu treffen.

## Aufgabe D

Für die Evaluation wurde der Macro F1-Score als primäres Entscheidungskriterium genommen. Mit 0.58 hat Gradient Boosting am besten abgeschnitten und wird damit für die Vorhersage des Benchmarkdatensatzes verwendet.
