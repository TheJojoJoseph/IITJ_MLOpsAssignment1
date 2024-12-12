from flask import Flask, request, render_template
import joblib
import pandas as pd
app = Flask(__name__)
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', feature_names=feature_names)


# prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from the form
        input_features = []
        for feature in feature_names:
            value = request.form.get(feature)
            if not value:
                raise ValueError(f"Value for '{feature}' is missing.")
            try:
                input_features.append(float(value))
            except ValueError:
                raise ValueError(f"Invalid value for '{
                                 feature}'. Please enter a numeric value.")

        input_data = pd.DataFrame(
            [input_features], columns=feature_names)  # Convert  DataFrame
        input_scaled = scaler.transform(input_data)   # Scale  input features
        prediction = model.predict(input_scaled)         # Make  prediction
        prediction = round(prediction[0], 2)
        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('index.html', error=str(e), feature_names=feature_names)


if __name__ == '__main__':
    app.run(debug=True)
