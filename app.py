from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model, scaler, and label encoders
model = pickle.load(open("salary_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

categorical_cols = ['workclass', 'gender', 'marital-status', 'occupation',
                    'relationship', 'race', 'native-country']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            "age": int(request.form['age']),
            "workclass": request.form['workclass'].strip(),
            "fnlwgt": int(request.form['fnlwgt']),
            "education-num": int(request.form['education-num']),
            "marital-status": request.form['marital-status'].strip(),
            "occupation": request.form['occupation'].strip(),
            "relationship": request.form['relationship'].strip(),
            "race": request.form['race'].strip(),
            "gender": request.form['gender'].strip(),
            "capital-gain": int(request.form['capital-gain']),
            "capital-loss": int(request.form['capital-loss']),
            "hours-per-week": int(request.form['hours-per-week']),
            "native-country": request.form['native-country'].strip()
        }

        # Encode categorical values
        for col in categorical_cols:
            try:
                input_data[col] = label_encoders[col].transform([input_data[col]])[0]
            except ValueError:
                return render_template('index.html', prediction_text=f"Invalid input: '{input_data[col]}' is not a recognized category for '{col}'")

        # Build feature array
        features = np.array([[input_data['age'], input_data['workclass'], input_data['fnlwgt'],
                              input_data['education-num'], input_data['marital-status'],
                              input_data['occupation'], input_data['relationship'],
                              input_data['race'], input_data['gender'], input_data['capital-gain'],
                              input_data['capital-loss'], input_data['hours-per-week'],
                              input_data['native-country']]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled_features)[0]
        result = ">50K" if prediction == 1 else "<=50K"

        return render_template("index.html", prediction_text=f"Predicted Income: {result}")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
