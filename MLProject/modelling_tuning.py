import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.sklearn

BASE_DIR = Path(__file__).parent
df = pd.read_csv(BASE_DIR / "diabetes_preprocessed.csv")

X = df.drop("diabetes", axis=1)
y = df["diabetes"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

param_grid = [
    {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10]},
    {"solver": ["lbfgs"], "penalty": ["l2"], "C": [0.01, 0.1, 1, 10]},
]

grid = GridSearchCV(
    LogisticRegression(max_iter=500, class_weight="balanced"),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
)

grid.fit(X_train, y_train)

best_model = grid.best_estimator_


mlflow.log_params(grid.best_params_)
mlflow.sklearn.log_model(best_model, "model")

y_pred = best_model.predict(X_test)

mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
