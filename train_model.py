import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle

# Load Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, delimiter=";")

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
