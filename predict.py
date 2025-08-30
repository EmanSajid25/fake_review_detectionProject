import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
with open('model/fake_review_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to predict review
def predict_review(review_text):
    cleaned_review = clean(review_text)  # Apply the same cleaning function
    vect_review = vectorizer.transform([cleaned_review])
    prediction = model.predict(vect_review)[0]
    return "Fake Review" if prediction == 1 else "Real Review"

# Test prediction
review = "This product is great! I love it."
result = predict_review(review)
print(result)  # Should print "Real Review" or "Fake Review"
