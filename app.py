import streamlit as st
import requests

# Page configuration
st.set_page_config(
    page_title="🕵️ Fake Review Detector",
    page_icon="🕵️",
    layout="centered"
)

# Custom styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem; 
        text-align: center;
        color: #00cec9;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        margin-bottom: 30px;
        color: #b2bec3;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .fake {
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        color: white;
    }
    .real {
        background: linear-gradient(to right, #11998e, #38ef7d);
        color: white;
    }
    .error {
        background: linear-gradient(to right, #f39c12, #e74c3c);
        color: white;
    }
    .stTextArea textarea {
        background-color: #2d3436;
        color: white;
        border-radius: 10px;
        padding: 15px;
    }
    .stButton>button {
        background: linear-gradient(to right, #00c9ff, #92fe9d);
        color: #2d3436;
        border: none;
        border-radius: 8px;
        font-weight: bold;
        padding: 10px 24px;
        font-size: 1rem;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .confidence {
        font-size: 1rem;
        margin-top: 10px;
        font-weight: normal;
        opacity: 0.8;
    }
    .server-info {
        margin-top: 30px;
        padding: 15px;
        background-color: #2d3436;
        border-radius: 10px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-title">🕵️ Fake Review Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect AI-generated fake product reviews</div>', unsafe_allow_html=True)

# Input area
review = st.text_area("📝 Enter product review", height=150, 
                     placeholder="Paste a product review here...",
                     help="Minimum 10 characters. Avoid code-like text")

# Prediction button
if st.button("🔍 Analyze Review", use_container_width=True):
    if not review.strip():
        st.warning("⚠️ Please enter a review")
    else:
        try:
            # Send request to FastAPI
            response = requests.post(
                "http://localhost:8000/predict", 
                json={"review": review},
                timeout=10
            )
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                
                if "prediction" in result:
                    if result["prediction"] == "Fake Review":
                        st.markdown(
                            f'<div class="result-box fake">🚨 FAKE REVIEW</div>'
                            f'<div class="confidence">Confidence: {result["confidence"]*100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="result-box real">✅ GENUINE REVIEW</div>'
                            f'<div class="confidence">Confidence: {result["confidence"]*100:.1f}%</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.error(f"Unexpected response: {result}")
                    
            elif response.status_code == 400:
                error_msg = response.json().get('error', 'Unknown error')
                st.markdown(
                    f'<div class="result-box error">❌ {error_msg}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.error(f"Server error ({response.status_code}): {response.text[:200]}")
                
        except requests.exceptions.ConnectionError:
            st.error("🔌 Could not connect to server. Please make sure the API server is running.")
        except requests.exceptions.Timeout:
            st.error("⌛ Request timed out. Please try again.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")

# Health check and server info
st.markdown("---")
with st.expander("Server Information"):
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=3)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API server is running")
            st.json(health_data)
        else:
            st.error(f"API server error: {health_response.text}")
    except:
        st.error("❌ API server is not reachable")

st.caption("Note: Make sure the FastAPI server is running at http://localhost:8000")
# ... [previous Streamlit code] ...

# Health check and server info
st.markdown("---")
with st.expander("Server Information"):
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=3)
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API server is running")
            
            # Display critical info
            cols = st.columns(2)
            cols[0].metric("Model Loaded", "Yes" if health_data["model_loaded"] else "No")
            cols[1].metric("Punkt Status", health_data["punkt_status"])
            
            st.write("NLTK Paths:")
            for path in health_data["nltk_paths"]:
                st.code(path)
        else:
            st.error(f"API server error: {health_response.text}")
    except:
        st.error("❌ API server is not reachable")

# Add resource verification
st.markdown("---")
st.subheader("Resource Verification")
if st.button("Verify NLTK Resources"):
    try:
        nltk_dir = os.path.join(os.getcwd(), "nltk_resources")
        punkt_path = os.path.join(nltk_dir, "tokenizers", "punkt", "PY3", "english.pickle")
        
        if os.path.exists(punkt_path):
            st.success(f"✅ Punkt found at: {punkt_path}")
            st.code(f"File size: {os.path.getsize(punkt_path) / 1024:.1f} KB")
        else:
            st.error(f"❌ Punkt not found at: {punkt_path}")
            
        # Show directory structure
        with st.expander("Show nltk_resources directory"):
            for root, dirs, files in os.walk(nltk_dir):
                st.write(f"📂 {root}")
                for file in files:
                    st.code(f"  - {file}")
    except Exception as e:
        st.error(f"Verification failed: {str(e)}")