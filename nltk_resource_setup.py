import os
import nltk
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_nltk():
    # Create a dedicated directory in the project
    nltk_dir = os.path.join(os.getcwd(), "nltk_resources")
    os.makedirs(nltk_dir, exist_ok=True)
    
    # Set NLTK to use this directory
    nltk.data.path.clear()
    nltk.data.path.append(nltk_dir)
    
    # Resources we need
    resources = {
        "punkt": ["tokenizers/punkt", "tokenizers/punkt/PY3/english.pickle"],
        "stopwords": ["corpora/stopwords", "corpora/stopwords/english"]
    }
    
    # Check and download each resource
    for resource, paths in resources.items():
        try:
            # Try to find the resource
            nltk.data.find(paths[0])
            logger.info(f"✅ Found {resource} in project resources")
        except LookupError:
            logger.warning(f"⬇️ Downloading {resource} to project directory...")
            try:
                # Download to project directory
                nltk.download(resource, download_dir=nltk_dir)
                
                # Verify download
                for path in paths:
                    if not nltk.data.find(path):
                        raise FileNotFoundError(f"Missing {path} after download")
                
                logger.info(f"✅ {resource} downloaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to download {resource}: {str(e)}")
                raise RuntimeError(f"Resource setup failed for {resource}")

if __name__ == "__main__":
    setup_nltk()
    print("NLTK resource setup complete!")
    print("Resource paths:", nltk.data.path)