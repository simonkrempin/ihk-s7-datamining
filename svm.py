from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from data_preperation import load_data
from data_visualization import display_result

data_columns, target_column = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    data_columns, target_column, test_size=0.2, stratify=target_column, random_state=42
)

svm_pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        (
            "svm",
            SVC(
                kernel="rbf",
                class_weight="balanced",
                C=1.0,
                probability=True,
                random_state=42,
            ),
        ),
    ]
)

svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)

display_result("SVM", y_test, y_pred)
