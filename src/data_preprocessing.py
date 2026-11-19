
from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

# Step 1: Load the Iris dataset
iris = load_iris(as_frame=True)

# Step 2: Convert dataset into a Pandas DataFrame
df = iris.frame

# Step 3: Convert numeric labels into flower names
df["species"] = df["target"].map(
    dict(enumerate(iris.target_names))
)

# Step 4: Remove the original numeric target column
df = df.drop(columns=["target"])

# Step 5: Define where to save the dataset
output_path = Path("data/raw/iris.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

# Step 6: Save dataset as CSV
df.to_csv(output_path, index=False)

# Step 7: Display results
print("Dataset created successfully!")
print("Total records:", len(df))
print("\nFirst five records:")
print(df.head())

print("\nDataset saved at:", output_path)
