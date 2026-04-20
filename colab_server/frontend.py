import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import io
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StyleMatch AI — Frontend",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS STYLING (Inspired by original app.py) ---
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    h1 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding-bottom: 10px;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
    }
    .upload-text {
        text-align: center;
        color: #BBBBBB;
        margin-bottom: 25px;
    }
    div.stImage {
        border-radius: 12px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }
    div.stImage:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.5);
        border: 1px solid #4ECDC4;
    }
    /* Card-like styling for the expander */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px;
    }
    /* Button styling */
    .stButton>button {
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        border-color: #4ECDC4;
        color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. CONFIGURATION & PATHS ---
# Points to the Local Docker Backend
LOCAL_BACKEND_URL = "http://localhost:8000/api/search"
COLAB_HEALTH_URL = "http://localhost:8000/health"

# Path to local images (one level up from colab_server/)
COLAB_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(COLAB_SERVER_DIR)
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")

# --- 4. HELPER FUNCTIONS ---
def get_image_path(article_id):
    """Finds image file in the local images/ directory."""
    subfolder = str(article_id)[:3]
    # Check multiple possible paths
    possible_paths = [
        os.path.join(IMAGES_DIR, subfolder, f"{article_id}.jpg"),
        os.path.join(PROJECT_ROOT, subfolder, f"{article_id}.jpg"),
        os.path.join(IMAGES_DIR, f"{article_id}.jpg")
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return p
    return None

def load_database():
    """Loads article IDs and metadata."""
    csv_path = os.path.join(PROJECT_ROOT, "articles.csv")
    ids_path = os.path.join(PROJECT_ROOT, "ids.npy")
    if not os.path.exists(ids_path) or not os.path.exists(csv_path):
        return None, None
    ids = np.load(ids_path, allow_pickle=True)
    df = pd.read_csv(csv_path, dtype={'article_id': str})
    return ids, df

item_ids, df = load_database()

def check_backend():
    """Checks if the local proxy and colab are online."""
    try:
        response = requests.get(COLAB_HEALTH_URL, timeout=3)
        if response.status_code == 200:
            return response.json()
    except:
        return None
    return None

# --- 5. UI LAYOUT ---
st.title("✨ StyleMatch AI")
st.subheader("Visual Fashion Recommender (Colab Backend)")

# Sidebar Status
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of results", 1, 10, 5)
    
    st.divider()
    st.header("📡 Connection Status")
    status = check_backend()
    if status:
        if status.get("colab_connection") == "ok":
            st.success("✅ Connected to Colab")
            st.info(f"Device: {status['colab_info'].get('device', 'unknown').upper()}")
        else:
            st.warning("⚠️ Local Proxy OK, but Colab is Offline")
    else:
        st.error("❌ Local Proxy (Docker) is Offline")
        st.caption("Run `docker-compose up` in colab_server folder.")

# Main Upload Section
# --- 6. EXPLORE & UPLOAD SECTION ---
st.markdown("<p class='upload-text'>Upload an image or pick an example from the gallery below!</p>", unsafe_allow_html=True)

# Gallery of random items for "Drag & Drop" or Quick Search
if item_ids is not None:
    with st.expander("🛍️ Explore Collection / Example Images", expanded=True):
        st.caption("Drag any of these images into the upload box, or click 'Use this' to search immediately.")
        example_cols = st.columns(5)
        
        # We use session state to refresh examples only when needed
        if 'example_indices' not in st.session_state:
            st.session_state['example_indices'] = np.random.choice(len(item_ids), 5, replace=False)
            
        if st.button("🔄 Refresh Examples"):
            st.session_state['example_indices'] = np.random.choice(len(item_ids), 5, replace=False)

        for i, idx in enumerate(st.session_state['example_indices']):
            rec_id = item_ids[idx]
            img_path = get_image_path(rec_id)
            if img_path:
                with example_cols[i]:
                    st.image(img_path, use_container_width=True)
                    if st.button(f"Use this", key=f"use_{rec_id}"):
                        st.session_state['selected_example'] = img_path
                        st.rerun()

# Main Upload Widget
uploaded_file = st.file_uploader("Upload your fashion item", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# Logic to determine which image to use
query_image = None
is_example = False

if uploaded_file is not None:
    query_image = Image.open(uploaded_file).convert('RGB')
    # Clear selected example if a new file is uploaded
    if 'selected_example' in st.session_state:
        del st.session_state['selected_example']
elif 'selected_example' in st.session_state:
    query_image = Image.open(st.session_state['selected_example']).convert('RGB')
    is_example = True

if query_image is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(query_image, caption="Current Query", use_container_width=True)
        if is_example:
            if st.button("❌ Clear Selection", use_container_width=True):
                del st.session_state['selected_example']
                st.rerun()
        
        search_btn = st.button("🔍 Find Similar Styles", use_container_width=True, type="primary")

    if search_btn:
        with st.spinner("Talking to Colab..."):
            img_bytes = io.BytesIO()
            query_image.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            
            try:
                files = {'image': ('query.jpg', img_bytes, 'image/jpeg')}
                data = {'top_k': top_k}
                
                response = requests.post(LOCAL_BACKEND_URL, files=files, data=data, timeout=60)
                
                if response.status_code == 200:
                    results_data = response.json()
                    results = results_data.get('results', [])
                    elapsed = results_data.get('elapsed_ms', 0)
                    
                    st.success(f"Found {len(results)} items in {elapsed}ms")
                    
                    # Display Results in a Grid
                    cols = st.columns(3)
                    for i, item in enumerate(results):
                        with cols[i % 3]:
                            rec_id = item['id']
                            score = item['score']
                            name = item['name']
                            
                            img_path = get_image_path(rec_id)
                            if img_path:
                                st.image(img_path, use_container_width=True)
                                st.markdown(f"**{name}**")
                                st.caption(f"Match Score: {int(score*100)}%")
                            else:
                                st.warning(f"Image Missing: {rec_id}")
                else:
                    st.error(f"Error from Backend: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to reach Local Backend: {e}")
else:
    st.info("Please upload an image or select an example from the gallery above to start.")
