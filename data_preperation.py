import pandas as pd


def load_data(file_path="DatenDataMiningAufgabe/DMAufgabeTrainingsdaten.csv"):
    df = pd.read_csv(
        file_path,
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

    # dropping the columns ANUMMER_01 to ANUMMER_10 and BESTELLIDENT as they are probably irrelevant to the task
    df = df.drop("ANUMMER_01", axis=1)
    df = df.drop("ANUMMER_02", axis=1)
    df = df.drop("ANUMMER_03", axis=1)
    df = df.drop("ANUMMER_04", axis=1)
    df = df.drop("ANUMMER_05", axis=1)
    df = df.drop("ANUMMER_06", axis=1)
    df = df.drop("ANUMMER_07", axis=1)
    df = df.drop("ANUMMER_08", axis=1)
    df = df.drop("ANUMMER_09", axis=1)
    df = df.drop("ANUMMER_10", axis=1)
    df = df.drop("BESTELLIDENT", axis=1)

    # convert categories into one hot encoding
    df = pd.get_dummies(
        df,
        columns=["Z_METHODE", "Z_CARD_ART", "TAG_BEST", "Z_LAST_NAME"],
        drop_first=True,
    )

    data_columns = None
    target_column = None

    if "TARGET_BETRUG" in df.columns:
        data_columns = df.drop("TARGET_BETRUG", axis=1)
        target_column = df["TARGET_BETRUG"]
    else:
        data_columns = df

    return data_columns, target_column
