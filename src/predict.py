
from pathlib import Path
import joblib
import pandas as pd

# Find our main project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Locate the saved ML model
model_path = BASE_DIR / "models" / "iris_model.pkl"

# Load the trained model
model = joblib.load(model_path)

print("=" * 40)
print("     SMART FLOWER PREDICTION SYSTEM")
print("=" * 40)

# Get measurements from the user
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

# Prepare the input data
input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ]
)

# Predict the flower species
prediction = model.predict(input_data)

# Calculate prediction probabilities
probabilities = model.predict_proba(input_data)

# Display result
print("\n" + "=" * 40)
print("PREDICTION RESULT")
print("=" * 40)

print("Predicted Species:", prediction[0])

print("\nPrediction Confidence:")

for species, probability in zip(model.classes_, probabilities[0]):
    print(f"{species}: {probability * 100:.2f}%")

print("=" * 40)
