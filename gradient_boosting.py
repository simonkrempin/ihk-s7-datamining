from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
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

gb_model = HistGradientBoostingClassifier(
    learning_rate=0.01,
    max_iter=1000,
    class_weight="balanced",
    max_leaf_nodes=64,
    early_stopping=False,
    random_state=42,
)

gb_model.fit(X_train, y_train)

joblib.dump(gb_model, "gb_model.joblib")

prediction = gb_model.predict(X_test)

display_result("Gradient Boosting", y_test, prediction)

# result = permutation_importance(gb_model, X_test, y_test, n_repeats=10, random_state=42)

# sorted_idx = result.importances_mean.argsort()

# print(sorted_idx)
