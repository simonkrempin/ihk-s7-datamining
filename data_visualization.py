from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd


def display_result(tool, correct, prediction):
    print(f"----- {tool} -----")

    labels = ["Normal", "Fraud"]
    cm = confusion_matrix(correct, prediction)
    df_cm = pd.DataFrame(
        cm, index=[f"Actual {l}" for l in labels], columns=[f"Pred {l}" for l in labels]
    )

    print(df_cm)

    print("\n")
    print(classification_report(correct, prediction, zero_division=0))
