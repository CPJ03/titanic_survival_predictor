from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return "Titanic API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.json
        
        # Extract age and sex from the request
        age = float(data['age'])
        sex = 1 if data['sex'].lower() == 'male' else 0  # Convert sex to binary (1 for male, 0 for female)
        
        # Create features array
        features = np.array([age, sex]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features).tolist()
        
        # Return prediction
        return jsonify({
            'status': 'success',
            'prediction': prediction[0],  # 1 means survived, 0 means did not survive
            'message': 'Congrats！You survived' if prediction[0] == 1 else 'Unfortunately you did not survive'
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)