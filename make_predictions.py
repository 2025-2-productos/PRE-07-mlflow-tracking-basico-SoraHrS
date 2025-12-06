import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

logged_model = "runs:/95f7bd995c0644d9af4c5f55ec5c0566/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
y = loaded_model.predict(x)
print(y)
