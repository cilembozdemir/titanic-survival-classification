"""Predict Titanic survival with clear, leakage-safe pipelines."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "titanic.csv"
OUTPUT_DIR = ROOT / "outputs"
RANDOM_STATE = 42


def make_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipeline, numeric_features),
         ("categorical", categorical_pipeline, categorical_features)]
    )


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    numeric = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical = ["Pclass", "Sex", "Embarked", "IsAlone"]
    X = data[numeric + categorical]
    y = data["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
        "Decision tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=RANDOM_STATE),
    }
    comparison = []
    fitted_models = {}
    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessing", make_preprocessor(numeric, categorical)), ("model", estimator)]
        )
        cv_auc = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc").mean()
        pipeline.fit(X_train, y_train)
        comparison.append({"model": name, "mean_cv_roc_auc": round(cv_auc, 3)})
        fitted_models[name] = pipeline

    comparison_df = pd.DataFrame(comparison).sort_values("mean_cv_roc_auc", ascending=False)
    comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = comparison_df.iloc[0]["model"]
    best_model = fitted_models[best_name]
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": int(len(data)),
        "selected_model": best_name,
        "holdout_accuracy": round(accuracy_score(y_test, predictions), 3),
        "holdout_f1": round(f1_score(y_test, predictions), 3),
        "holdout_roc_auc": round(roc_auc_score(y_test, probabilities), 3),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Titanic survival - {best_name}")
    plt.xlabel("Predicted survival")
    plt.ylabel("Actual survival")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    survival_by_group = data.groupby(["Sex", "Pclass"], as_index=False)["Survived"].mean()
    survival_by_group.to_csv(OUTPUT_DIR / "survival_by_sex_and_class.csv", index=False)
    print(comparison_df.to_string(index=False))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
