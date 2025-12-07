from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

mapping_yes_no = {"ja": 1, "nein": 0}

df = pd.read_csv("DatenDataMiningAufgabe/DMAufgabeTrainingsdaten.csv", sep=";")

# convert yes/no to 1/0 for sklearn to correclty generate the tree
df["TARGET_BETRUG"] = df["TARGET_BETRUG"].map(mapping_yes_no)
df["B_EMAIL"] = df["B_EMAIL"].map(mapping_yes_no)
df["B_TELEFON"] = df["B_TELEFON"].map(mapping_yes_no)
df["FLAG_LRIDENTISCH"] = df["FLAG_LRIDENTISCH"].map(mapping_yes_no)
df["FLAG_NEWSLETTER"] = df["FLAG_NEWSLETTER"].map(mapping_yes_no)
df["Z_LAST_NAME"] = df["Z_LAST_NAME"].map(mapping_yes_no)
df["CHK_LADR"] = df["CHK_LADR"].map(mapping_yes_no)
df["CHK_RADR"] = df["CHK_RADR"].map(mapping_yes_no)
df["CHK_KTO"] = df["CHK_KTO"].map(mapping_yes_no)
df["CHK_CARD"] = df["CHK_CARD"].map(mapping_yes_no)
df["CHK_COOKIE"] = df["CHK_COOKIE"].map(mapping_yes_no)
df["CHK_IP"] = df["CHK_IP"].map(mapping_yes_no)
df["FAIL_LPLZ"] = df["FAIL_LPLZ"].map(mapping_yes_no)
df["FAIL_LPLZORTMATCH"] = df["FAIL_LPLZORTMATCH"].map(mapping_yes_no)
df["FAIL_RPLZ"] = df["FAIL_RPLZ"].map(mapping_yes_no)
df["FAIL_RORT"] = df["FAIL_RORT"].map(mapping_yes_no)
df["FAIL_RPLZORTMATCH"] = df["FAIL_RPLZORTMATCH"].map(mapping_yes_no)
df["NEUKUNDE"] = df["NEUKUNDE"].map(mapping_yes_no)

# convert the string date into processable date values
df["B_GEBDATUM"] = pd.to_datetime(df["B_GEBDATUM"])
df["TIME_BEST"] = pd.to_datetime(df["TIME_BEST"])
df["DATUM_LBEST"] = pd.to_datetime(df["DATUM_LBEST"])

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
print(confusion_matrix(y_test, prediction))
print("\n")
print(classification_report(y_test, prediction))
