import subprocess
subprocess.call(["pip", "install", "fsspec", "s3fs"])

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


input_path = "s3://wine-quality-dataset-680/winequality-red.csv"

# Load dataset
data = pd.read_csv(input_path, delimiter=";")

# Preprocessing
X = data.drop(columns=["quality"])
y = data["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Save Model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)
