
import pandas as pd
import matplotlib.pyplot as plt

# STEP 1: Load the dataset
df = pd.read_csv("data/raw/iris.csv")

# STEP 2: Display the first five records
print("\nFIRST FIVE RECORDS")
print(df.head())

# STEP 3: Check dataset size
print("\nDATASET SHAPE")
print(df.shape)

# STEP 4: Understand column information
print("\nDATASET INFORMATION")
df.info()

# STEP 5: Statistical summary
print("\nSTATISTICAL SUMMARY")
print(df.describe())

# STEP 6: Check missing values
print("\nMISSING VALUES")
print(df.isnull().sum())

# STEP 7: Count flower species
print("\nFLOWER SPECIES COUNT")
print(df["species"].value_counts())

# STEP 8: Create a graph
df["species"].value_counts().plot(kind="bar")

plt.title("Number of Flowers by Species")
plt.xlabel("Flower Species")
plt.ylabel("Number of Records")

plt.tight_layout()
plt.show()
