from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('rf_hprf.joblib')  # Ensure this points to the correct model file

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect and prepare feature data
    feature_values = [
        request.form['S1_Temp'],
        request.form['S2_Temp'],
        request.form['S3_Temp'],
        request.form['S4_Temp'],
        request.form['S1_Light'],
        request.form['S2_Light'],
        request.form['S3_Light'],
        request.form['S4_Light'],
        request.form['S1_Sound'],
        request.form['S2_Sound'],
        request.form['S3_Sound'],
        request.form['S4_Sound'],
        request.form['S5_CO2'],
        request.form['S5_CO2_Slope'],
        request.form['S6_PIR'],
        request.form['S7_PIR']
    ]
    features = np.array([float(i) for i in feature_values]).reshape(1, -1)
    
    # Apply the same preprocessing as was done for training data, if any

    # Make prediction
    prediction = model.predict(features)[0]  # assuming model expects array
    
    # Return result
    return render_template('index.html', prediction_text=f'Estimated number of people: {prediction}')

if __name__ == '__main__':
    app.run(debug=True)
