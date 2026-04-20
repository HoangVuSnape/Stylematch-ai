import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import io
from PIL import Image

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="StyleMatch AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. CUSTOM CSS STYLING ---
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Header Styling */
    h1 {
        color: #FFFFFF;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        text-align: center;
        padding-bottom: 20px;
        background: -webkit-linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Card Styling for Results */
    div.stImage {
        border-radius: 10px;
        transition: transform 0.3s;
    }
    div.stImage:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Upload Section */
    .upload-text {
        text-align: center;
        color: #FAFAFA;
        font-size: 1.2rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. PATH SETUP ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "articles.csv")
IDS_PATH = os.path.join(BASE_DIR, "ids.npy")

# Backend API server address (chạy trong Notebook cell)
BACKEND_URL = "http://localhost:5001"

# --- 4. LOAD LIGHTWEIGHT DATA (No model needed!) ---
@st.cache_data
def load_database():
    if not os.path.exists(IDS_PATH) or not os.path.exists(CSV_PATH):
        return None, None
    ids = np.load(IDS_PATH, allow_pickle=True)
    df = pd.read_csv(CSV_PATH, dtype={'article_id': str})
    return ids, df

item_ids, df = load_database()

# --- 5. LOGIC: Gọi API Backend để tìm ảnh tương tự ---
def get_image_path(article_id):
    subfolder = article_id[:3]
    path_nested = os.path.join(BASE_DIR, "images", subfolder, article_id + ".jpg")
    path_root = os.path.join(BASE_DIR, subfolder, article_id + ".jpg")
    if os.path.exists(path_nested): return path_nested
    if os.path.exists(path_root): return path_root
    return None

def find_similar_items(image, top_k=5):
    """Gửi ảnh tới Backend API server (đang chạy ở Notebook cell) để inference."""
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)

    try:
        response = requests.post(
            f"{BACKEND_URL}/search",
            files={'image': ('query.jpg', img_bytes, 'image/jpeg')},
            data={'top_k': top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()  # list of {index, id, score, name}
    except requests.exceptions.ConnectionError:
        st.error("❌ Không kết nối được Backend! Hãy chắc chắn đã chạy Cell 'Load Model & Start Backend' trên Colab trước.")
        return None
    except Exception as e:
        st.error(f"❌ Lỗi gọi Backend: {e}")
        return None

# --- 6. THE UI LAYOUT ---

# Header Section
st.markdown("<h1>✨ StyleMatch AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #BBBBBB; margin-bottom: 40px;'>Visual Search Engine powered by OpenAI CLIP</p>", unsafe_allow_html=True)

# Main Container
if item_ids is None:
    st.error("⚠️ Database missing! Check repo files (ids.npy, articles.csv).")
else:
    # Sidebar for Upload
    with st.sidebar:
        st.header("📸 Start Here")
        uploaded_file = st.file_uploader("Upload an item", type=["jpg", "png", "jpeg"])
        
        st.markdown("---")
        st.markdown("### 💡 How it works")
        st.info(
            "This app doesn't use simple tags. "
            "It uses **Computer Vision** (CLIP) to understand "
            "texture, shape, and style to find "
            "visually similar matches."
        )
        st.markdown("---")
        st.caption("Model chạy trên Colab Backend (Cell riêng)")

    # Main Content Area
    if uploaded_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Query")
            user_image = Image.open(uploaded_file).convert("RGB")
            st.image(user_image, use_container_width=True, caption="Analyzed Pattern")
            
            if st.button("🔍 Search Database", type="primary", use_container_width=True):
                with st.spinner("🧠 Đang gửi ảnh tới AI Backend để phân tích..."):
                    results = find_similar_items(user_image)
                    if results is not None:
                        st.session_state['results'] = results

        with col2:
            st.subheader("Top Recommendations")
            if 'results' in st.session_state and st.session_state['results']:
                results = st.session_state['results']
                
                # Show results in a clean grid - Row 1 (3 items)
                cols = st.columns(3)
                for i, item in enumerate(results[:3]):
                    with cols[i]:
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

                # Second row for remaining items
                if len(results) > 3:
                    cols_2 = st.columns(3)
                    for i, item in enumerate(results[3:5]):
                        with cols_2[i]:
                            rec_id = item['id']
                            score = item['score']
                            name = item['name']
                            
                            img_path = get_image_path(rec_id)
                            if img_path:
                                st.image(img_path, use_container_width=True)
                                st.markdown(f"**{name}**")
                                st.caption(f"Match Score: {int(score*100)}%")
            else:
                st.info("👈 Upload an image in the sidebar to begin.")
    
    else:
        # Welcome State
        st.markdown("<div class='upload-text'>waiting for input...</div>", unsafe_allow_html=True)
        st.subheader("Explore the Collection")
        example_cols = st.columns(5)
        random_indices = np.random.choice(len(item_ids), 5)
        for i, idx in enumerate(random_indices):
            with example_cols[i]:
                rec_id = item_ids[idx]
                path = get_image_path(rec_id)
                if path: st.image(path, use_container_width=True)
