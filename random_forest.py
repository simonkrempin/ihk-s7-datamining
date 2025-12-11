from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from data_preperation import load_data
from data_visualization import display_result
import joblib

data_columns, target_column = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    data_columns,
    target_column,
    test_size=0.2,
    random_state=42,
    stratify=target_column,  # this is to maintain the ratio of the data
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

rf_model.fit(X_train, y_train)

joblib.dump(rf_model, "rf_model.joblib")

prediction = rf_model.predict(X_test)

display_result("Random Forest", y_test, prediction)
