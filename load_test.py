import pickle

# Load the saved model and vectorizer
model = pickle.load(open("model/fake_review_model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

print("✅ Model and vectorizer successfully loaded.")
