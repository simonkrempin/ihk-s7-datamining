from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

df = pd.read_csv(
    "DatenDataMiningAufgabe/DMAufgabeTrainingsdaten.csv",
    sep=";",
    decimal=",",
    true_values=["ja"],
    false_values=["nein"],
)

# convert the string date into processable date values
df["B_GEBDATUM"] = pd.to_datetime(df["B_GEBDATUM"])
df["B_GEBDATUM_YEAR"] = df["B_GEBDATUM"].dt.year
df["B_GEBDATUM_MONTH"] = df["B_GEBDATUM"].dt.month
df["B_GEBDATUM_DAY"] = df["B_GEBDATUM"].dt.day
df = df.drop("B_GEBDATUM", axis=1)

df["TIME_BEST"] = pd.to_datetime(df["TIME_BEST"])
df["TIME_BEST_HOUR"] = df["TIME_BEST"].dt.hour
df = df.drop("TIME_BEST", axis=1)

df["DATUM_LBEST"] = pd.to_datetime(df["DATUM_LBEST"])
df["DATUM_LBEST_YEAR"] = df["DATUM_LBEST"].dt.year
df["DATUM_LBEST_MONTH"] = df["DATUM_LBEST"].dt.month
df["DATUM_LBEST_DAY"] = df["DATUM_LBEST"].dt.day
df = df.drop("DATUM_LBEST", axis=1)

# convert categories into one hot encoding
df = pd.get_dummies(
    df, columns=["Z_METHODE", "Z_CARD_ART", "TAG_BEST"], drop_first=True
)

data_columns = df.drop("TARGET_BETRUG", axis=1)
target_column = df["TARGET_BETRUG"]

X_train, X_test, y_train, y_test = train_test_split(
    data_columns,
    target_column,
    test_size=0.2,
    random_state=42,
    stratify=target_column,
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

rf_model.fit(X_train, y_train)

prediction = rf_model.predict(X_test)

print("----- Random Forest -----")

labels = ["Normal", "Fraud"]
cm = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(
    cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]
)

print(df_cm)

print("\n")
print(classification_report(y_test, prediction, zero_division=0))
