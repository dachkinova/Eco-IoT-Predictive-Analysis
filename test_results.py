import tkinter as tk
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

def test_bagging_regressor(base_estimator_class):
    results_text.delete(1.0, tk.END)
    data = pd.read_csv("merged_data.csv")
    X = data[['AirTemp', 'Press', 'UMR']]
    y = data[['NO', 'NO2', 'O3', 'PM10']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = RobustScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_estimators_list = [5, 10, 15]
    max_samples_list = [0.5, 0.7, 1.0]
    max_features_list = [0.5, 0.7, 1.0]

    base_estimator = base_estimator_class()
    
    for n_estimators in n_estimators_list:
        for max_samples in max_samples_list:
            for max_features in max_features_list:
                bagging_regressor = BaggingRegressor(
                    estimator=base_estimator,
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    max_features=max_features,
                    n_jobs=4,
                    random_state=42
                )
                bagging_regressor.fit(X_train_scaled, y_train)
                score = bagging_regressor.score(X_test_scaled, y_test)

                result_text = (f"Base Estimator: {base_estimator_class.__name__}, "
                               f"n_estimators: {n_estimators}, "
                               f"max_samples: {max_samples}, "
                               f"max_features: {max_features}, "
                               f"Score: {score}\n")
                results_text.insert(tk.END, result_text)


root = tk.Tk()
root.title("Model Comparison")

bagging_regressors = [RandomForestRegressor, DecisionTreeRegressor, LinearRegression]

for base_estimator in bagging_regressors:
    button_text = f"Test {base_estimator.__name__}"
    button = tk.Button(root, text=button_text, command=lambda est=base_estimator: test_bagging_regressor(est))
    button.pack()

results_text = tk.Text(root, height=20, width=100)
results_text.pack()

root.mainloop()


