import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("data/student.csv")
X = df.drop("final_gpa", axis=1)
y = df["final_gpa"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestRegressor()
model.fit(X_train, y_train)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Model saved successfully!")
from sklearn.metrics import r2_score, mean_absolute_error

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("R² Score:", round(r2_score(y_test, predictions), 3))
print("MAE:", round(mean_absolute_error(y_test, predictions), 3))