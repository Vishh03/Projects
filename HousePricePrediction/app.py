from flask import Flask, render_template, request, jsonify
import joblib
from pandas import DataFrame
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

from xgboost import XGBRegressor

# loading the XGBoost Regression model
regressor2 = XGBRegressor()

# fitting the model with training data
regressor2.fit(X_train, Y_train)

# prediction on training data
training_data_prediction = regressor2.predict(X_train)

# R squared value

r2_train = metrics.r2_score(Y_train, training_data_prediction)
"""print("R squared vale : ", r2_train)"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Predict on the testing data
y_pred = regressor2.predict(X_test)

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, y_pred)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(Y_test, y_pred)

# Calculate R-squared (R2) score
r2 = r2_score(Y_test, y_pred)
regressor3 = XGBRegressor()  # Define the regressor3 variable
regressor3.fit(X_train, Y_train)

a = df[["SALES_PRICE"]]  # Numerical data for price
b = df["AREA"]  # Categorical data for area

# Encode categorical data using LabelEncoder
le = LabelEncoder()
b = le.fit_transform(b)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(a, b)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        INT_SQFT = request.form["INT_SQFT"]
        DIST_MAINROAD = request.form["DIST_MAINROAD"]
        N_BEDROOM = request.form["N_BEDROOM"]
        N_BATHROOM = request.form["N_BATHROOM"]
        N_ROOM = request.form["N_ROOM"]
        QS_ROOMS = request.form["QS_ROOMS"]
        QS_BATHROOM = request.form["QS_BATHROOM"]
        QS_BEDROOM = request.form["QS_BEDROOM"]
        QS_OVERALL = request.form["QS_OVERALL"]
        REG_FEE = request.form["REG_FEE"]
        COMMIS = request.form["COMMIS"]
        PARK_FACIL = request.form["PARK_FACIL"]
        features = np.array(
            [
                float(INT_SQFT),
                float(DIST_MAINROAD),
                float(N_BEDROOM),
                float(N_BATHROOM),
                float(N_ROOM),
                float(QS_ROOMS),
                float(QS_BATHROOM),
                float(QS_BEDROOM),
                float(QS_OVERALL),
                float(REG_FEE),
                float(COMMIS),
                float(PARK_FACIL),
            ]
        )

        input_data_reshaped = features.reshape(1, -1)
        prediction = regressor3.predict(input_data_reshaped)
        output = prediction
        price = prediction[0]
        """prediction = model.predict([features])
        data = {}
        for feature in features:
            data[feature] = float(request.form[feature])
            
            try:
                data[feature] = float(request.form[feature])
            except KeyError:
                return f"Missing form field: {feature}", 400

        prediction = model.predict(pd.DataFrame(data, index=[0]))"""
        # output = prediction[0]
        return render_template("index.html", prediction=price, show_result=True)
        # return jsonify({"Predicted Sales Price": output})
    return render_template("index.html", show_result=False)


@app.route("/predict_area", methods=["GET", "POST"])
def predict_area():
    if request.method == "POST":
        start_price = float(request.form["start_price"])
        end_price = float(request.form["end_price"])

        # Predict the area based on the input price range
        predicted_area = le.inverse_transform(clf.predict([[start_price]]))[0]
        # return f"The predicted area for the price range ${start_price} to ${end_price} is {predicted_area}"
        return render_template(
            "predict_area.html", prediction=predicted_area, show_result=True
        )
    return render_template("predict_area.html", show_result=False)


if __name__ == "__main__":
    app.run(debug=True)
