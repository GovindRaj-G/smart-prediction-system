
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# STEP 1: Locate project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# STEP 2: Load dataset
data_path = BASE_DIR / "data" / "raw" / "iris.csv"
df = pd.read_csv(data_path)

print("Original dataset:", df.shape)

# STEP 3: Separate features (X) and target (y)
X = df.drop(columns=["species"])
y = df["species"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())

# STEP 4: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# STEP 5: Create processed data folder
output_path = BASE_DIR / "data" / "processed"
output_path.mkdir(parents=True, exist_ok=True)

# STEP 6: Save training and testing data
X_train.to_csv(output_path / "X_train.csv", index=False)
X_test.to_csv(output_path / "X_test.csv", index=False)

y_train.to_csv(output_path / "y_train.csv", index=False)
y_test.to_csv(output_path / "y_test.csv", index=False)

# STEP 7: Display results
print("\nDATA PREPARATION COMPLETED!")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining data saved successfully!")
print("Testing data saved successfully!")
