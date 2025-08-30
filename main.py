import os
import re
import pickle
import logging
import nltk
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import requests
import zipfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- NLTK RESOURCE SETUP ----
def ensure_punkt_resources():
    """Ensure all Punkt resources are available with multiple fallback strategies"""
    try:
        # First try standard method
        nltk.data.find('tokenizers/punkt')
        logger.info("✅ Punkt resources found via standard nltk.data")
        return True
    except LookupError:
        logger.warning("⚠️ Punkt not found via standard method")
    
    # Create a dedicated directory for NLTK resources
    nltk_dir = os.path.join(os.getcwd(), "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    nltk.data.path.append(nltk_dir)
    logger.info(f"➕ Added custom NLTK path: {nltk_dir}")
    
    # Define resource locations
    punkt_path = os.path.join(nltk_dir, "tokenizers", "punkt")
    
    # Check if punkt exists in our custom directory
    if os.path.exists(os.path.join(punkt_path, "PY3", "english.pickle")):
        logger.info("✅ Punkt found in custom directory")
        return True
    
    # Download punkt if still missing
    logger.warning("⬇️ Downloading punkt resources...")
    try:
        nltk.download('punkt', download_dir=nltk_dir)
        logger.info("✅ Punkt downloaded to custom directory")
        return True
    except Exception as e:
        logger.error(f"❌ Punkt download failed: {str(e)}")
    
    # Ultimate fallback: manually download from GitHub
    logger.warning("⬇️ Attempting manual download from GitHub...")
    try:
        # Create necessary directories
        os.makedirs(punkt_path, exist_ok=True)
        
        # Download punkt resources
        url = "https://github.com/nltk/nltk_data/raw/gh-pages/packages/tokenizers/punkt.zip"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save and extract
        zip_path = os.path.join(punkt_path, "punkt.zip")
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(punkt_path)
        
        os.remove(zip_path)
        logger.info("✅ Punkt downloaded manually from GitHub")
        return True
    except Exception as e:
        logger.error(f"❌ Manual download failed: {str(e)}")
        return False

# Initialize NLTK resources
if not ensure_punkt_resources():
    logger.critical("❌ CRITICAL: Failed to load Punkt resources after all attempts")
    raise RuntimeError("Punkt tokenizer not available")

# Verify punkt is loaded
try:
    nltk.data.find('tokenizers/punkt')
    logger.info("✅ Punkt tokenizer ready")
except LookupError as e:
    logger.critical(f"❌ CRITICAL: Punkt not found after all attempts: {str(e)}")
    raise RuntimeError("Punkt tokenizer not available")

# ---- FASTAPI APP ----
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- MODEL LOADING ----
try:
    model_path = os.path.join(os.getcwd(), "model", "fake_review_model.pkl")
    vectorizer_path = os.path.join(os.getcwd(), "model", "vectorizer.pkl")
    
    logger.info(f"🔍 Loading model from {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    logger.info(f"🔍 Loading vectorizer from {vectorizer_path}")
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

# ---- TEXT PROCESSING ----
try:
    nltk.download('stopwords', quiet=True)
    stopwords_set = set(nltk.corpus.stopwords.words('english'))
    logger.info("✅ Stopwords loaded successfully")
except LookupError:
    logger.warning("⚠️ Stopwords not found, using minimal set")
    stopwords_set = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
                    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 
                    'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 
                    'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 
                    'their', 'theirs', 'themselves'}

def clean_text(text):
    """Clean and preprocess review text"""
    if not isinstance(text, str) or len(text.strip()) < 10:
        return ""
    
    # Filter code-like patterns
    if re.search(r'\b(import|def|function|class|return|\.py|print|for |if )\b', text):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    
    # Use nltk word tokenization
    try:
        words = nltk.word_tokenize(text)
    except Exception as e:
        logger.error(f"⚠️ Tokenization failed: {str(e)}")
        # Fallback to simple tokenization
        words = text.split()
    
    words = [w for w in words if w not in stopwords_set and len(w) > 2]
    return ' '.join(words)

# ---- API ENDPOINTS ----
class ReviewInput(BaseModel):
    review: str

@app.post("/predict")
def predict_review(input: ReviewInput):
    try:
        review = input.review
        logger.info(f"📥 Received review: {review[:50]}...")
        
        cleaned = clean_text(review)
        if not cleaned:
            return {"error": "Invalid review. Too short or contains code patterns."}
        
        vect = vectorizer.transform([cleaned])
        pred = model.predict(vect)[0]
        proba = model.predict_proba(vect)[0]
        confidence = max(proba)
        
        result = "Fake Review" if pred == 1 else "Real Review"
        logger.info(f"🔮 Prediction: {result} (Confidence: {confidence:.2f})")
        
        return {
            "input": review,
            "prediction": result,
            "confidence": float(confidence)
        }
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    punkt_status = "Available"
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        punkt_status = "Not Available"
    
    return {
        "status": "OK", 
        "model_loaded": model is not None,
        "punkt_status": punkt_status,
        "nltk_paths": nltk.data.path
    }

@app.get("/debug/punkt")
def debug_punkt():
    """Endpoint to debug punkt resource issues"""
    results = []
    for path in nltk.data.path:
        punkt_dir = os.path.join(path, "tokenizers", "punkt")
        exists = os.path.exists(punkt_dir)
        results.append({
            "path": punkt_dir,
            "exists": exists,
            "contents": os.listdir(punkt_dir) if exists else []
        })
    return results