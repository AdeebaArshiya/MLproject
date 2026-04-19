from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load AI model as a background helper
model = None
try:
    if os.path.exists('eeg_model.pkl'):
        model = joblib.load('eeg_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('label_encoder.pkl')
except Exception as e:
    print("AI model not found, using Expert Rules mode.")

def predict_emotion(alpha, beta, gamma):
    try:
        a, b, g = float(alpha), float(beta), float(gamma)

        # 1. PRIMARY RULE: ALPHA DOMINANCE (CALM)
        # If Alpha is the highest, it is calm regardless of the scale
        if a > b and a > g:
            return 'calm'
        
        # 2. SECONDARY RULE: BETA DOMINANCE (STRESS)
        # If Beta is significantly higher than Alpha
        if b > a and b > g:
            return 'Stress'
        
        # 3. TERTIARY RULE: GAMMA SPIKE (EXCITED)
        # If Gamma is the leading wave
        if g > a and g > b:
            return 'Excited'

        # FALLBACK TO AI MODEL (Only if rules are tied)
        if model:
            # Scale large inputs down to tiny decimals for the SVM model
            scaled_data = pd.DataFrame([[a/10000, b/10000, g/10000]], 
                                       columns=['mean_1_a', 'mean_2_a', 'mean_3_a'])
            input_scaled = scaler.transform(scaled_data)
            pred = model.predict(input_scaled)
            return le.inverse_transform(pred)[0]

    except:
        return 'calm'
    return 'calm'

@app.route('/')
def home():
    return render_template('index.html', alpha=0, beta=0, gamma=0)

@app.route('/predict', methods=['POST'])
def predict():
    a = request.form.get('alpha', 0)
    b = request.form.get('beta', 0)
    g = request.form.get('gamma', 0)
    result = predict_emotion(a, b, g)
    return render_template('index.html', prediction_text=f"State: {result}", alpha=a, beta=b, gamma=g)

@app.route('/predict_live', methods=['POST'])
def predict_live():
    data = request.get_json()
    res = predict_emotion(data['alpha'], data['beta'], data['gamma'])
    return jsonify({'prediction': res})

if __name__ == '__main__':
    app.run(debug=True)