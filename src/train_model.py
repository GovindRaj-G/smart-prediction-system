
from pathlib import Path

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# 1. Find the main project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# 2. Load the prepared training and testing data
data_path = BASE_DIR / "data" / "processed"

X_train = pd.read_csv(data_path / "X_train.csv")
X_test = pd.read_csv(data_path / "X_test.csv")

y_train = pd.read_csv(data_path / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(data_path / "y_test.csv").squeeze("columns")


# 3. Create the Machine Learning model
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])


# 4. Train the model
model.fit(X_train, y_train)

print("MODEL TRAINING COMPLETED!")


# 5. Make predictions using testing data
y_pred = model.predict(X_test)


# 6. Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)

print("\nMODEL ACCURACY:")
print(f"{accuracy * 100:.2f}%")


print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))


print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))


# 7. Save the trained model
model_path = BASE_DIR / "models"
model_path.mkdir(parents=True, exist_ok=True)

joblib.dump(model, model_path / "iris_model.pkl")

print("\nTRAINED MODEL SAVED SUCCESSFULLY!")
print("Location:", model_path / "iris_model.pkl")
