import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('punkt')

def clean(text):
    if not isinstance(text, str):
        return ""

    # Remove code-like patterns
    if any(keyword in text for keyword in ['import', 'def', '{', '}', '()', 'function', '=', ';']):
        return ""

    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Filter too short reviews or non-natural text
    if len(words) < 4:
        return ""

    return ' '.join(words)

def preprocess_dataset(path):
    df = pd.read_csv(path)
    df['label'] = df['label'].map({'CG': 1, 'OR': 0})
    df['clean_text'] = df['text_'].apply(clean)
    df = df[df['clean_text'] != '']  # Remove empty or invalid reviews
    X = df['clean_text']
    y = df['label']
    return train_test_split(X, y, test_size=0.2, random_state=42)
