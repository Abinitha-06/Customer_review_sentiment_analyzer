from flask import Flask, render_template, request
import pickle
import nltk
import re
from sklearn.linear_model import LogisticRegression
import csv
import os
from datetime import datetime
import time

# Optional matplotlib for multiple-review pie chart rendering.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except Exception:
    has_matplotlib = False
    plt = None

app = Flask(__name__)

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Load the trained model and vectorizer
try:
    with open('sentiment_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    print("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Please run model.py first to train the model.")
    model = None
    vectorizer = None

# Preprocessing function (same as in model.py)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
    return ' '.join(tokens)

# Function to save review to CSV
def save_review_to_csv(review, sentiment, confidence):
    file_path = 'review_log.csv'
    file_exists = os.path.exists(file_path)
    with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Review", "Sentiment", "Confidence"])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, review, sentiment, f"{confidence}%"])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if model is None or vectorizer is None:
        return render_template('index.html', error='Model not loaded. Please train the model first.')

    reviews_text = request.form.get('reviews', '').strip()
    if not reviews_text:
        return render_template('index.html', error='Please enter at least one review to analyze.')

    # Split by newlines and filter empty lines
    review_lines = [line.strip() for line in reviews_text.splitlines() if line.strip()]

    if len(review_lines) == 0:
        return render_template('index.html', error='Please enter at least one review to analyze.')

    # Analyze each review
    results = []
    predictions = []
    
    for review_text in review_lines:
        if not review_text:
            continue
            
        # Preprocess the input
        cleaned_review = preprocess_text(review_text)

        # Vectorize the input
        review_vectorized = vectorizer.transform([cleaned_review])

        # Make prediction
        prediction = model.predict(review_vectorized)[0]

        # Get confidence scores
        probabilities = model.predict_proba(review_vectorized)[0]
        confidence = round(max(probabilities) * 100, 1)

        # Add to results
        results.append({
            'review': review_text,
            'prediction': prediction,
            'confidence': confidence
        })

        predictions.append(prediction)

        # Save the review to CSV
        save_review_to_csv(review_text, prediction, confidence)

    # Calculate statistics
    total = len(results)
    positive_count = predictions.count('positive')
    negative_count = predictions.count('negative')
    neutral_count = predictions.count('neutral')

    positive_percent = round(positive_count / total * 100, 1) if total > 0 else 0
    negative_percent = round(negative_count / total * 100, 1) if total > 0 else 0
    neutral_percent = round(neutral_count / total * 100, 1) if total > 0 else 0

    # Determine mode
    mode = 'single' if len(review_lines) == 1 else 'multiple'

    # Get prediction and confidence for single review mode
    single_prediction = results[0]['prediction'] if len(results) == 1 else None
    single_confidence = results[0]['confidence'] if len(results) == 1 else None

    # Cache busting timestamp for pie chart
    chart_time = int(time.time())

    # If multiple reviews, generate matplotlib pie chart
    if mode == 'multiple' and has_matplotlib and plt is not None:
        chart_path = os.path.join(app.root_path, 'static', 'pie_chart.png')
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [positive_count, neutral_count, negative_count]
        colors = ['#2a9d8f', '#457b9d', '#e76f51']

        plt.figure(figsize=(5, 5), dpi=100)
        wedges, texts, autotexts = plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
            textprops={'color': '#20314d', 'fontsize': 11, 'fontweight': '600'}
        )
        plt.setp(autotexts, size=10, weight='bold', color='white')
        plt.title('Sentiment distribution', fontsize=14, color='#1f3f63', pad=14)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(chart_path, transparent=True, bbox_inches='tight', dpi=100)
        plt.close()

    return render_template(
        'index.html',
        results=results,
        mode=mode,
        prediction=single_prediction,
        confidence=single_confidence,
        positive_percent=positive_percent,
        negative_percent=negative_percent,
        neutral_percent=neutral_percent,
        total_reviews=total,
        positive_count=positive_count,
        negative_count=negative_count,
        neutral_count=neutral_count,
        chart_time=chart_time,
        has_matplotlib=has_matplotlib,
    )

if __name__ == '__main__':
    app.run(debug=True)
