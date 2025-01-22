from flask import Flask, request, jsonify, render_template
import joblib

# Load the trained model
model_path = 'P:/Aswath/final/RDF_Project/cyberbullying_detection_model.pkl'
model = joblib.load(model_path)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    # Render the 'index.html' template located in the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"message": "Invalid input. Please provide a 'text' field."}), 400

        text = data.get('text', '')

        if not text.strip():
            return jsonify({"message": "Input text cannot be empty."}), 400

        prediction = model.predict([text])[0]
        return jsonify({"result": prediction})

    except Exception as e:
        return jsonify({"message": "An error occurred.", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)
