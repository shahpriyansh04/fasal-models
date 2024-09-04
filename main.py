from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

ensemble_model = joblib.load('ensemble_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the Crop Prediction API!"

@app.route('/predict', methods=['GET'])
def predict():
        N = float(request.args.get('N'))
        P = float(request.args.get('P'))
        K = float(request.args.get('K'))
        temperature = float(request.args.get('temperature'))
        humidity = float(request.args.get('humidity'))
        ph = float(request.args.get('ph'))
        rainfall = float(request.args.get('rainfall'))
        
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        input_data_scaled = scaler.transform(input_data)
        
        prediction = ensemble_model.predict(input_data_scaled)
        
        return jsonify({'prediction': prediction[0]})
    
   
if __name__ == '__main__':
    app.run(debug=True)