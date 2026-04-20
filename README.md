# 🛍️ StyleMatch AI: Visual-Semantic Fashion Recommender

> **A "Visual Search" engine that recommends clothing based on style similarity, not just text tags.**
> *Powered by OpenAI CLIP, PyTorch, and Streamlit.*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://stylematch-ai-app.streamlit.app/)
[![View Code](https://img.shields.io/badge/View%20Code-GitHub-black?logo=github)](https://github.com/Muhammad-Shahan/stylematch-ai)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![OpenAI CLIP](https://img.shields.io/badge/Model-OpenAI%20CLIP-green)
![Status](https://img.shields.io/badge/Status-Live-success)

---

## 🚀 Try It Live
The application is deployed and ready to test!  
**👉 [Click here to launch StyleMatch AI](https://stylematch-ai-app.streamlit.app/)**

---

## 📚 Technical Docs

Structured internal documentation is available in [`docs/`](docs/):

1. [`00_Project_Overview.md`](docs/00_Project_Overview.md)
2. [`01_Repository_Contents.md`](docs/01_Repository_Contents.md)
3. [`02_End_to_End_Workflow.md`](docs/02_End_to_End_Workflow.md)
4. [`03_Data_and_Embedding_Pipeline.md`](docs/03_Data_and_Embedding_Pipeline.md)
5. [`04_Operations_and_Next_Steps.md`](docs/04_Operations_and_Next_Steps.md)

---

## 📖 The Problem: Why "Traditional" isn't enough?

Most traditional recommendation systems (like those used by Amazon or Netflix 10 years ago) rely on **Collaborative Filtering**—analyzing who bought what. While powerful, this approach has a major flaw:

* ❌ **The Cold Start Problem:** If a new shirt is released today, nobody has bought it yet. The system has no data, so it **never recommends it**.
* ❌ **Keyword Limits:** Searching for "Blue Shirt" returns thousands of results, but it doesn't understand *style* (e.g., "Bohemian", "Minimalist", or "Streetwear").

## 💡 The Solution: Visual-Semantic Search

**StyleMatch AI** ignores purchase history. Instead, it uses **Computer Vision** to "see" the product just like a human stylist would.

It encodes the **visual style** (texture, pattern, cut, shape) into a high-dimensional vector space. If you upload a photo of a *floral summer dress*, the AI finds other items with a similar *vibe*, even if they are in different categories or have never been purchased together.

### 🌟 Key Features
* **🧠 Multimodal Intelligence:** Uses OpenAI's **CLIP** (Contrastive Language-Image Pre-training) to bridge the gap between images and meaning.
* **🔍 Vector Similarity Engine:** Performs lightning-fast Nearest Neighbor search using **Cosine Similarity** on 512-dimensional vectors.
* **⚡ Zero-Shot Capability:** Instantly recommends brand-new items without needing any user history.
* **☁️ Edge-Optimized:** Architected to run purely on CPU (via Streamlit Cloud), proving that powerful AI doesn't always need expensive GPUs for inference.

---

## 📐 System Architecture

The project is built in two distinct phases to ensure scalability and speed:



### Phase 1: Offline Ingestion (The "Brain")
1.  **Ingestion:** We process a dataset of 5,000+ fashion images (H&M Dataset).
2.  **Embedding:** Each image is passed through the **CLIP Vision Encoder**.
3.  **Vectorization:** The model outputs a **512-dimensional embedding** (a list of numbers representing the image's style).
4.  **Indexing:** These vectors are normalized and stored in `.npy` files for ultra-fast retrieval.

### Phase 2: Online Inference (The App)
1.  **User Input:** You upload a photo (e.g., your own shirt).
2.  **Real-Time Encoding:** The app converts your image into a query vector using CLIP.
3.  **Similarity Search:** It calculates the mathematical distance (dot product) between your image and the 5,000 database items.
4.  **Retrieval:** The system returns the Top 5 most visually similar items in < 1 second.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python-based Web Framework)
* **AI Model:** `openai/clip-vit-base-patch32` (Vision Transformer)
* **Deep Learning:** PyTorch, HuggingFace Transformers
* **Data Processing:** Pandas, NumPy, Pillow
* **Dataset:** H&M Personalized Fashion Recommendations (Resized 256x256)

---

## 💻 Local Installation

If you want to run this project on your own machine:

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/Muhammad-Shahan/stylematch-ai.git](https://github.com/Muhammad-Shahan/stylematch-ai.git)
    cd stylematch-ai
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 🔮 Future Roadmap

* **Hybrid Filtering:** Combine this visual engine with user transaction history for higher accuracy.
* **Text-to-Image Search:** Allow users to search by typing (e.g., *"Red dress for a wedding"*) using CLIP's text encoder.
* **Scalability:** Migrate from local files to a Vector Database (Pinecone/Milvus) to handle millions of items.

---

## 👨‍💻 Author

**Built by Muhammad Shahan** 

If you found this project interesting, feel free to connect!

[![LinkedIn](https://www.linkedin.com/in/m-shahan/) [![GitHub](https://github.com/Muhamad-Shahan/)
