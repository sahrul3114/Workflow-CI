import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

# Load dataset
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop("diabetes", axis=1)
y = df["diabetes"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = [
    {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100]},
    {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.01, 0.1, 1, 10, 100]},
    {"solver": ["saga"], "penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10, 100]},
    {
        "solver": ["saga"],
        "penalty": ["elasticnet"],
        "l1_ratio": [0, 0.5, 1],
        "C": [0.01, 0.1, 1, 10, 100],
    },
]

grid = GridSearchCV(
    LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=42,
    ),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# SET EXPERIMENT (WAJIB)
mlflow.set_experiment("Skilled_Modelling_Tuning")

mlflow.log_params(grid.best_params_)

mlflow.sklearn.log_model(
    best_model,
    artifact_path="model"
)

y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.log_metric("accuracy", acc)
mlflow.log_metric("f1_score", f1)

print("Accuracy:", acc)
print("F1 score:", f1)
