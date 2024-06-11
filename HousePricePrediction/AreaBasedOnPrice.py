import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv("Chennai houseing sale.csv")

# Separate features (X) and target variable (y)
X = data[["SALES_PRICE"]]  # Numerical data for price
y = data["AREA"]  # Categorical data for area

# Encode categorical data using LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X, y)

# Get input from the user for house price range
start_price = float(input("Enter the starting price range: "))
end_price = float(input("Enter the ending price range: "))

# Predict the area based on the input price range
predicted_area = le.inverse_transform(clf.predict([[start_price]]))[0]
print(
    f"The predicted area for the price range ${start_price} to ${end_price} is {predicted_area}"
)
