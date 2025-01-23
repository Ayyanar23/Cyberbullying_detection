from flask import Flask, request, jsonify, render_template
import joblib
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

# Load the trained model
model_path = 'Y:/Project/Cybe/final/RDF_Project/cyberbullying_detection_model.pkl'
model = joblib.load(model_path)

# Initialize the Flask app
app = Flask(__name__)

# Path to the CSV file
CSV_FILE = 'comments.csv'

# Ensure the CSV file exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Comment', 'Prediction'])  # Write headers

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle comment prediction and store it in the CSV file."""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"message": "Invalid input. Please provide a 'text' field."}), 400

        text = data.get('text', '')

        if not text.strip():
            return jsonify({"message": "Input text cannot be empty."}), 400

        # Predict using the trained model
        prediction = model.predict([text])[0]

        # Save the comment and prediction in the CSV file
        with open(CSV_FILE, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([text, prediction])

        return jsonify({"result": prediction})

    except Exception as e:
        return jsonify({"message": "An error occurred.", "error": str(e)}), 500

@app.route('/analysis')
def analysis():
    """Render the data visualization page."""
    if not os.path.exists(CSV_FILE):
        return "No data available for analysis.", 404

    # Load data from CSV
    data = pd.read_csv(CSV_FILE)

    # Generate a bar chart for prediction counts
    pred_counts = data['Prediction'].value_counts()
    plt.figure(figsize=(6, 4))
    pred_counts.plot(kind='bar', color=['#007BFF', '#FF5733', '#28A745', '#FFC107'])
    plt.title('Prediction Counts')
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.tight_layout()

    # Save the plot as a PNG image in memory
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()

    return render_template('analysis.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)
