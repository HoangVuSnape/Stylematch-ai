# 🚀 StyleMatch AI - Colab Model Server & Local Proxy

This project allows you to host heavy AI models (CLIP) on **Google Colab** (leveraging free GPUs) while interacting with them from your **Local Machine** via a Dockerized backend.

## 🏗️ Architecture
1. **Colab Server (`app_colab.py`)**: Runs the CLIP model, handles image embedding, and performs vector search. Exposed via `ngrok`.
2. **Local Proxy (`app_local.py`)**: A Flask gateway running in Docker that forwards requests from your frontend to the Colab URL.
3. **Discord Integration**: Real-time logs for health checks and search results.

---

## 🛠️ Setup Instructions

### Part 1: Google Colab (The Model Server)
1. Open Google Colab and enable **T4 GPU** (`Runtime` > `Change runtime type`).
2. Clone your repository or upload the following files to `/content/`:
   - `app_colab.py`
   - `Colab_Setup.ipynb`
   - Your data: `articles.csv`, `embeddings.npy`, `ids.npy` (usually inside a folder).
3. Open and run **`Colab_Setup.ipynb`**:
   - Install dependencies.
   - Enter your **Ngrok Authtoken** (get it from [ngrok.com](https://dashboard.ngrok.com/)).
   - Run the final cell to start the server.
4. **Copy the public URL** generated (e.g., `https://xxxx-xxxx.ngrok-free.app`).

### Part 2: Local Machine (The Proxy)
1. Go to the `colab_server` directory.
2. Create or edit the **`.env`** file:
   ```env
   COLAB_URL=https://your-ngrok-url.ngrok-free.app
   PORT=8000
   ```
3. Start the Docker container:
   ```bash
   docker compose up --build
   ```

### Part 3: Streamlit Frontend (Optional)
1. Run the frontend locally (ensure you have `streamlit` and `requests` installed):
   ```bash
   streamlit run frontend.py
   ```
2. Access the UI at `http://localhost:8501`.

---

## 📡 API Endpoints

### 1. Health Check
Check if the local proxy and the Colab server are connected.
- **URL**: `GET http://localhost:8000/health`
- **Response**:
  ```json
  {
    "local_status": "online",
    "colab_connection": "ok",
    "colab_info": { "device": "cuda", "items": 10000, ... }
  }
  ```

### 2. Image Search
Send an image to get similar items.
- **URL**: `POST http://localhost:8000/api/search`
- **Body (form-data)**: 
  - `image`: [File]
  - `top_k`: 5 (optional)
- **Response**: List of matches with scores and metadata.

---

## 🔔 Monitoring
- All search activities and server status are sent to the configured **Discord Webhook**.
- You can also view raw logs at `GET http://localhost:8000/logs` (proxied from Colab).

## 📝 Note on Colab Usage
- Colab sessions expire after some time. If the connection fails, restart the Colab notebook and update the `COLAB_URL` in your local `.env` file.
- Ensure the `root` path in `app_colab.py` matches your file structure on Colab.
