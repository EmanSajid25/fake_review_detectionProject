import pandas as pd
import os
import pickle
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Download required resources
nltk.download('stopwords')
nltk.download('punkt')

# Unified cleaning function
def clean_text(text):
    if not isinstance(text, str) or len(text.strip()) < 10:
        return ""
    
    # Filter code-like patterns
    if re.search(r'\b(import|def|function|class|return|\.py|print|for |if )\b', text):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = nltk.word_tokenize(text)
    
    stop_words = set(nltk.corpus.stopwords.words('english'))
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def preprocess_dataset(path):
    # Read dataset
    df = pd.read_csv(path)
    
    # Handle label mapping
    df['label'] = df['label'].map({'CG': 1, 'OR': 0})
    df = df.dropna(subset=['label'])
    
    # Clean text
    df['clean_text'] = df['text_'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 10]  # Remove short reviews
    
    return df

def train_model():
    # Windows path with raw string
    df = preprocess_dataset(r"C:\Users\aiman\OneDrive\Desktop\fake_review_detector\data\fake_review.csv")
    
    # Check class distribution
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    # Split data
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorization
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words='english'
    )
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        C=0.5,
        solver='liblinear',
        random_state=42
    )
    model.fit(X_train_vect, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vect)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save artifacts
    os.makedirs("model", exist_ok=True)
    with open("model/fake_review_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("model/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print("✅ Model trained and saved successfully")

if __name__ == "__main__":
    train_model()