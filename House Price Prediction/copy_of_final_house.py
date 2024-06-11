import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib

# loading the data from csv file to a Pandas DataFrame
df = pd.read_csv(r"Chennai houseing sale.csv")

ddf = df.drop(
    [
        "PRT_ID",
        "AREA",
        "DATE_SALE",
        "UTILITY_AVAIL",
        "STREET",
        "MZZONE",
        "SALE_COND",
        "DATE_BUILD",
        "BUILDTYPE",
    ],
    axis=1,
)

ddf.isnull().sum()

ddf.dropna(inplace=True)

ddf.isnull().sum()

ddf.replace({"PARK_FACIL": {"No": 0, "Yes": 1, "Noo": 2}}, inplace=True)

X = ddf.drop(columns="SALES_PRICE", axis=1)
Y = ddf["SALES_PRICE"]

print(X.to_string())

print(Y.to_string())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

from xgboost import XGBRegressor

# loading the XGBoost Regression model
regressor2 = XGBRegressor()

# fitting the model with training data
regressor2.fit(X_train, Y_train)

# prediction on training data
training_data_prediction = regressor2.predict(X_train)

# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print("R squared vale : ", r2_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on the testing data
y_pred = regressor2.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(Y_test, y_pred)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

input_1 = []
sqft = int(input("Enter the INT_SQFT"))
input_1.append(sqft)
dist = int(input("Enter the DIST_MAINROAD"))
input_1.append(dist)
bed = int(input("Enter the N_BEDROOM"))
input_1.append(bed)
bath = int(input("Enter the N_BATHROOM"))
input_1.append(bath)
room = int(input("Enter the N_ROOM"))
input_1.append(room)


qroom = int(input("Enter the QS_ROOMS"))
input_1.append(qroom)
qbath = int(input("Enter the QS_BATHROOM"))
input_1.append(qbath)
qbed = int(input("Enter the QS_BEDROOM"))
input_1.append(qbed)
all = int(input("Enter the QS_OVERALL"))
input_1.append(all)
fee = int(input("Enter the REG_FEE"))
input_1.append(fee)
com = int(input("Enter the COMMIS"))
input_1.append(com)
park = bool(input("Enter the PARK_FACIL(0-No,1-Yes)"))
input_1.append(park)


input_data = tuple(input_1)
# changing input_data to a numpy array
regressor3 = XGBRegressor()  # Define the regressor3 variable
regressor3.fit(X_train, Y_train)  # Fit the model with training data

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = regressor3.predict(input_data_reshaped)
print(prediction)

# print('The insurance cost is USD ', prediction[0])
# Taking prediction value to USD for change it to INR
Price = prediction[0]

print("The currency in INR is", round(Price, 1))
joblib.dump(regressor2, "xgboost_model.pkl")
