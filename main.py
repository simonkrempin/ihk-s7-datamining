import joblib
from data_preperation import load_data
from data_visualization import display_result
import questionary
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

model_selection = questionary.select(
    "Welches Model soll genutzt werden",
    choices=["Random Forest", "Gradient Boosting", "Support Vector Machine"],
).ask()

data_columns, target = load_data(
    "DatenDataMiningAufgabe/DMAufgabeKlassifizierungsdaten.csv"
)

model = None
match model_selection:
    case "Random Forest":
        model = joblib.load("rf_model.joblib")
    case "Gradient Boosting":
        model = joblib.load("gb_model.joblib")
    case "Support Vector Machine":
        model = joblib.load("svm_model.joblib")
    case _:
        raise Exception("Model not found")

prediction = model.predict(data_columns)

plt.figure(figsize=(8, 5))
ax = sns.countplot(x=prediction)
plt.title("Verteilung der Vorhersagen (Predicted Classes)")
plt.xlabel("Klasse")
plt.ylabel("Anzahl")
ax.set_xticklabels(["Normal", "Fraud"])
plt.show()
