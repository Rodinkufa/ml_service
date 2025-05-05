# app/model.py

import os
import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

AVAILABLE_MODELS = {
    "logistic": LogisticRegression(max_iter=200),
    "random_forest": RandomForestClassifier(n_estimators=100)
}
current_model_name = "logistic"

def train_and_save_model(name: str = "logistic", params: dict = None):
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.3, random_state=42
    )

    if params is None:
        params = {}

    model_class = AVAILABLE_MODELS[name].__class__
    model = model_class(**params)
    
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.joblib"))
    with open(os.path.join(MODEL_DIR, f"{name}_accuracy.txt"), "w") as f:
        f.write(str(acc))




def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.joblib")
    if not os.path.exists(path):
        train_and_save_model(name)
    return joblib.load(path)

def get_available_models():
    return list(AVAILABLE_MODELS.keys())

def get_model_accuracy(name: str):
    path = os.path.join(MODEL_DIR, f"{name}_accuracy.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            return float(f.read())
    return None
