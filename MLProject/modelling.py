import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--random_state", type=int, default=42)
args = parser.parse_args()



# Aktifkan Autolog sebelum proses training
mlflow.sklearn.autolog()

# Load Dataset
# Pastikan file ini ada di folder yang sama dengan modelling.py saat di-push
df = pd.read_csv("Titanic_Dataset_Analysis_preprocessing.csv")

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=args.random_state
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Set Experiment
mlflow.set_experiment("Titanic Random Forest")

# MLflow Project otomatis akan mencatat ke run yang sedang aktif
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=args.random_state
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
mlflow.log_metric("accuracy_manual", accuracy)

print(f"Accuracy: {accuracy:.2f}")
