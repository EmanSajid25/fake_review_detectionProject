import nltk

def download_nltk_resources():
    resources = ['punkt', 'stopwords']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
            print(f"✅ {resource} already installed")
        except LookupError:
            print(f"⬇️ Downloading {resource}...")
            nltk.download(resource)
            print(f"✅ {resource} downloaded successfully")

if __name__ == "__main__":
    download_nltk_resources()