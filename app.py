from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

# Load the vectorizer and model
loaded_vectorizer = joblib.load(r'C:\Users\HD Gallions\OneDrive\Desktop\Tester\tfidf_vectorizer.joblib')
loaded_model = joblib.load(r'C:\Users\HD Gallions\OneDrive\Desktop\Tester\pac_model.joblib')

# Function to classify text
def classify_text(text):
    # Transform the input text using the loaded vectorizer
    tfidf_text = loaded_vectorizer.transform([text])
    # Use the loaded model to make predictions
    prediction = loaded_model.predict(tfidf_text)
    return "Fake" if prediction[0] == 1 else "Real"

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['text']
        # Get the text from the form and classify it
        prediction = classify_text(news_text)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
