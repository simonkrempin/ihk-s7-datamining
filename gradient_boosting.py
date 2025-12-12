from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from data_preperation import load_data
from data_visualization import display_result
import joblib

# we tried with native categories but one hot encoding resulted in improved performance / otherwise you could use
# the native categories -> resulted in a worse performance
data_columns, target_column = load_data()
# data_columns, target_column = load_data(one_hot_encoding=False)

category_mask = data_columns.dtypes == "category"

X_train, X_test, y_train, y_test = train_test_split(
    data_columns,
    target_column,
    test_size=0.2,
    random_state=42,
    stratify=target_column,  # this is to maintain the ratio of the data
)

gb_model = HistGradientBoostingClassifier(
    learning_rate=0.01,  # learning_rate less then 0.01 results in underfitting
    max_iter=1000,  # maximum number of trees; ensemble complexity
    class_weight="balanced",  # balanced for consideration of unbalanced data -> adjustes the weight inversily proportional
    max_leaf_nodes=64,
    early_stopping=False,  # early stopping true -> degression, because likely a temporary drop in performance
    random_state=42,  # reproducability
    l2_regularization=1.0,  # reduces outlier confidence
    categorical_features=category_mask,  # can get ignored
)

gb_model.fit(X_train, y_train)

joblib.dump(gb_model, "gb_model.joblib")

prediction = gb_model.predict(X_test)

display_result("Gradient Boosting", y_test, prediction)

# result = permutation_importance(gb_model, X_test, y_test, n_repeats=10, random_state=42)

# sorted_idx = result.importances_mean.argsort()

# print(sorted_idx)
