import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

#Load dataset
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "diabetes_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

# Split features & target
X = df.drop("diabetes", axis=1)
y = df["diabetes"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow setup
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Basic_Modelling")

# Train model
with mlflow.start_run():
    mlflow.sklearn.autolog()

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)

    print("Accuracy:", acc)
    print("F1 score:", f1)