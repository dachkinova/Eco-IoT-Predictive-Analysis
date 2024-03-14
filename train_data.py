import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib as joblib
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression

csv = pandas.read_csv("merged_data.csv")
x = csv[['AirTemp', 'Press', 'UMR']]
y = csv[['NO', 'NO2', 'O3', 'PM10']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
scaler = RobustScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

joblib.dump(scaler, 'scaler.joblib')

bagging_models = []

for i in range(y_train.shape[1]):
    bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(),
                                     n_estimators=10,
                                     n_jobs=4,
                                     random_state=0)
    bagging_regressor.fit(x_train_scaled, y_train.iloc[:, i])
    bagging_models.append(bagging_regressor)

    joblib.dump(bagging_models[i], 'bagging_regressor_' + y.columns[i] + '.joblib')

scores = [bagging_models[i].score(x_test_scaled, y_test.iloc[:, i]) for i in range(len(bagging_models))]
print("Model Scores:", scores)

arr = np.array([[25, 955, 80]])
arr_scaled = scaler.transform(arr)

predictions = [bagging_models[i].predict(arr_scaled) for i in range(len(bagging_models))]
print("Predictions:", predictions)
