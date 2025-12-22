import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn

# Tracking lokal
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")

# Autolog
mlflow.sklearn.autolog()

# Load Dataset
df = pd.read_csv('preprocessing/Titanic_Dataset_Analysis_preprocessing.csv')

X = df.drop('Survived', axis=1)
y = df['Survived']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Experiment
mlflow.set_experiment("Titanic Random Forest")

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Metric tambahan (opsional)
    mlflow.log_metric("accuracy_manual", accuracy)

    print(f"Accuracy: {accuracy:.2f}")


