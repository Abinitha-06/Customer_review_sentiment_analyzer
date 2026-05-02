import pandas as pd
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import pickle
import random

# Download necessary NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Generate synthetic dataset
def generate_synthetic_reviews(num_reviews=1500):
    positive_words = [
        'amazing', 'excellent', 'fantastic', 'great', 'wonderful', 'awesome', 'brilliant',
        'outstanding', 'perfect', 'superb', 'incredible', 'marvelous', 'fabulous', 'terrific',
        'love', 'loved', 'best', 'highly recommend', 'must have', 'worth every penny',
        'quality', 'reliable', 'durable', 'easy to use', 'user friendly', 'efficient',
        'fast', 'quick', 'convenient', 'comfortable', 'sleek', 'modern', 'innovative'
    ]

    negative_words = [
        'terrible', 'awful', 'horrible', 'worst', 'disappointing', 'poor', 'bad', 'hate',
        'hated', 'waste', 'useless', 'broken', 'defective', 'cheap', 'low quality',
        'unreliable', 'difficult', 'complicated', 'slow', 'uncomfortable', 'ugly',
        'outdated', 'frustrating', 'annoying', 'expensive for what it is', 'overpriced'
    ]

    neutral_words = [
        'okay', 'fine', 'average', 'decent', 'acceptable', 'standard', 'normal',
        'regular', 'ordinary', 'typical', 'basic', 'simple', 'plain', 'mediocre',
        'not bad', 'not great', 'so so', 'alright', 'fair', 'reasonable', 'adequate'
    ]

    reviews = []
    sentiments = []

    # Generate positive reviews
    for _ in range(num_reviews // 3):
        num_words = random.randint(3, 8)
        words = random.sample(positive_words, num_words)
        review = ' '.join(words) + '.'
        reviews.append(review)
        sentiments.append('positive')

    # Generate negative reviews
    for _ in range(num_reviews // 3):
        num_words = random.randint(3, 8)
        words = random.sample(negative_words, num_words)
        review = ' '.join(words) + '.'
        reviews.append(review)
        sentiments.append('negative')

    # Generate neutral reviews
    for _ in range(num_reviews - 2 * (num_reviews // 3)):
        num_words = random.randint(3, 8)
        words = random.sample(neutral_words, num_words)
        review = ' '.join(words) + '.'
        reviews.append(review)
        sentiments.append('neutral')

    return reviews, sentiments

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str):  # Ensure the text is a string
        return ""  # Return empty string if not a valid text
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    stopwords = nltk.corpus.stopwords.words('english')  # Remove stopwords
    tokens = nltk.word_tokenize(text)  # Tokenize
    tokens = [word for word in tokens if word.isalnum() and word not in stopwords]
    return ' '.join(tokens)

# Generate synthetic data
print("Generating synthetic dataset...")
reviews, sentiments = generate_synthetic_reviews(1500)

# Create DataFrame
data = pd.DataFrame({
    'Review': reviews,
    'Sentiment': sentiments
})

# Apply preprocessing to the reviews
print("Preprocessing text...")
data['cleaned_review'] = data['Review'].apply(preprocess_text)

# Split data into features (X) and labels (y)
X = data['cleaned_review']
y = data['Sentiment']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Fit vectorizer on training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save the trained model and vectorizer
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model trained and saved as sentiment_model.pkl")
print("Vectorizer saved as vectorizer.pkl")
print(f"Dataset size: {len(data)} reviews")
print(f"Training set: {len(X_train)} reviews")
print(f"Test set: {len(X_test)} reviews")
print(f"Sentiment distribution: {data['Sentiment'].value_counts().to_dict()}")
