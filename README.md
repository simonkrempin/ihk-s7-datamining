# Aufgaben Teil 1

## Aufgabe A

Die Daten folgen einer ungleichen Verteilung. Insgesamt liegen von 30000 Datenpunkten nur 1746 Betrugsfälle vor, was lediglich 5,82% ergeben und somit ein stark inbalancierter Datensatz vorliegt. Relevant ist dabei, dass während der Messung die Imbalancen passend addressiert werden. Im Mittelpunkt dessen steht ein adequates Messinstrument wie der Macro F1-Score, der einen Fokus auf die Klassengewichtung setzt.

## Aufgabe B

Die Spalten ANUMMER\_01 bis ANUMMMER\_10 repräsentieren die Artikelnummern der Bestellungen. Die Repräsentation der Artikel über eine Liste ist dahingehend ungeeignet, da zwei kritische Probleme damit einher gehen. Erstens sind die Bestellungen damit auf 10 begrenzt und Warenkörbe ab 11 Elementen können nicht akurat dargestellt werden. Dazu kommt, dass eine hohe Anzahl an NULL werten gespeichert werden muss. Ein Vorteil einer solchen Darstellung ist eine verbesserte Performance der Query, da kein Leistung auf die JOIN-Operation aufgewandt wird. Eine abgewandte Darstellung, indem die Arikel in eine weitere 1-n Tabelle ausgelagert werden, ermöglicht eine flexiblere Handhabung der Artikel.

## Aufgabe C

Für die Implementation wurde das Python Framework Scikit-Learn verwendet. Die Algorithem dafür belaufen sich auf Random Forest, Gradient Boosting und Support Vector Machine. Random Forest sind eine Sammlung von Decision Trees, die zusammen die Vorhersage ermitteln. Ähnlich dazu nutzt Gradient Boosting aneinandergereite
Decision Trees um die Präzision zu verbessenn. Zuguterletzt nutzt Support Vector Machine eine lineare Regression, um die Vorhersage zu ermitteln.

Für

## Aufgabe D

Für die Evaluation wurden die F1-Score und die AUC-Score verwendet. Der höchste Macro F1-Score wurde von Gradient Boosting mit x erzielt.
