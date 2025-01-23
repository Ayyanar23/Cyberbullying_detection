from flask import Flask, request, jsonify, render_template_string
import joblib

# Load the trained model
model_path = 'P:/Aswath/final/RDF_Project/cyberbullying_detection_model.pkl'
model = joblib.load(model_path)

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """
    Renders the home page with an input form.
    """
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"message": "Invalid input. Please provide a 'text' field."}), 400

        text = data.get('text', '')
        if not text.strip():
            return jsonify({"message": "Input text cannot be empty."}), 400

        # Log input text for debugging
        print(f"Input text: {text}")

        # Make a prediction
        prediction = model.predict([text])[0]

        # Log prediction
        print(f"Prediction result: {prediction}")

        return jsonify({"result": prediction})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"message": "An error occurred.", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)
