from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Load model and encoder
model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'kmeans_model.pkl')
encoder_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'gender_encoder.pkl')

with open(model_path, 'rb') as f:
    kmeans_model = pickle.load(f)

with open(encoder_path, 'rb') as f:
    gender_encoder = pickle.load(f)

# Mapping cluster to description
cluster_descriptions = {
    0: "Low income, low spending",
    1: "High income, high spending",
    2: "Low income, high spending",
    3: "High income, low spending",
    4: "Middle income, balanced spending"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    age = int(request.form['age'])
    income = float(request.form['income'])
    score = float(request.form['score'])

    gender_encoded = gender_encoder.transform([[gender]])[0]

    input_data = pd.DataFrame([[gender_encoded, age, income, score]],
                              columns=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    cluster = int(kmeans_model.predict(input_data)[0])
    description = cluster_descriptions.get(cluster, "Unknown Segment")

    return render_template('result.html', segment=cluster, description=description)

if __name__ == '__main__':
    app.run(debug=True)
