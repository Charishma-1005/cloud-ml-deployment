# Step 1: Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Step 2: Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 5: Save the model to disk
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print(" Model trained and saved as model.pkl")