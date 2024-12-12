import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib


data = pd.read_csv('boston.csv')

X = data.drop('MV', axis=1)  # Separate  (X) and target (y)
y = data['MV']
X = X[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
       'AGE', 'DIS', 'RAD', 'TAX', 'PT', 'B', 'LSTAT']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)  # Split data into training and testing sets

scaler = StandardScaler()  # Scale data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training
model = LinearRegression()
model.fit(X_train_scaled, y_train)
# Save the scaler and model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'linear_regression_model.pkl')


print("Model and scaler  complete")
