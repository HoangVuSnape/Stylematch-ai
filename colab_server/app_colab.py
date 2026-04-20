import numpy as np
import pandas as pd
import torch
import time
from transformers import CLIPProcessor, CLIPModel
from flask import Flask, request, jsonify
from PIL import Image
import io
import logging
import threading

# Because i just clone repo
root= "/content/Stylematch-ai"

# --- Configuration ---
MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDINGS_FILE = root + "/embeddings.npy"
IDS_FILE = root + "/ids.npy"
METADATA_FILE = root + "/articles.csv"
DISCORD_WEBHOOK = "https://discord.com/api/webhooks/1491031437288280146/B5xnLJ_qxU_nxL63GnzUmSvhCHGMw9Dxv5SzxQLyBJVs0V-Y2Jf22y2svv6DLiqPRti5"

# --- Global State for Logging ---
search_logs = []
total_searches = 0

# --- Helper functions ---
def send_discord(content=None, embed=None):
    if not DISCORD_WEBHOOK: return
    payload = {}
    if content: payload['content'] = content
    if embed: payload['embeds'] = [embed]
    try:
        import requests
        requests.post(DISCORD_WEBHOOK, json=payload, timeout=5)
    except:
        pass

# --- Model & Data Loading ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("⏳ Loading CLIP Model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
logger.info(f"✅ CLIP Model loaded on {device}")

logger.info("⏳ Loading database...")
try:
    embeddings = np.load(EMBEDDINGS_FILE).astype(np.float32)
    ids = np.load(IDS_FILE, allow_pickle=True)
    df = pd.read_csv(METADATA_FILE, dtype={'article_id': str})
    logger.info(f"✅ Database loaded: {len(ids)} items")
except Exception as e:
    logger.error(f"❌ Error loading database: {e}")
    embeddings, ids, df = None, None, None

# --- Flask App ---
app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "device": device,
        "items": len(ids) if ids is not None else 0,
        "total_searches": total_searches
    })

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify(search_logs)

@app.route('/search', methods=['POST'])
def search():
    if embeddings is None:
        return jsonify({"error": "Database not loaded"}), 500
        
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
        
    start_time = time.time()
    file = request.files['image']
    top_k = int(request.form.get('top_k', 5))
    
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # Inference
        inputs = processor(images=image, return_tensors='pt', padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            
            # Robustness: Handle cases where CLIP might return an output object instead of a raw tensor
            if not isinstance(image_features, torch.Tensor):
                image_features = image_features.pooler_output if hasattr(image_features, 'pooler_output') else image_features[0]
                
                # Apply visual projection if it exists and dimension doesn't match expected CLIP embedding size
                if hasattr(model, 'visual_projection') and image_features.shape[-1] != 512:
                    image_features = model.visual_projection(image_features)

            # Normalize
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = image_features.cpu().numpy()

        # Similarity Search (Cosine Similarity via dot product on normalized vectors)
        scores = np.dot(query_vector, embeddings.T).flatten()
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            article_id = str(ids[idx])
            meta = df[df['article_id'] == article_id]
            name = meta.iloc[0].get('prod_name', 'Unknown') if not meta.empty else 'Unknown'
            results.append({
                'id': article_id,
                'score': float(scores[idx]),
                'name': name,
                'index': int(idx)
            })

        elapsed = time.time() - start_time
        
        # Logging
        global total_searches
        total_searches += 1
        log_entry = {
            'time': time.strftime('%H:%M:%S'),
            'search_no': total_searches,
            'top_result': results[0]['name'] if results else 'None',
            'top_score': round(results[0]['score'] * 100, 1) if results else 0,
            'elapsed_ms': int(elapsed * 1000)
        }
        search_logs.append(log_entry)
        
        # Send to Discord (Async via Thread to not block response)
        msg = f"[{log_entry['time']}] 🔍 Search #{total_searches} | Top: {log_entry['top_result']} ({log_entry['top_score']}%) | {log_entry['elapsed_ms']}ms"
        threading.Thread(target=send_discord, kwargs={'content': msg}).start()

        return jsonify({
            "results": results,
            "elapsed_ms": int(elapsed * 1000)
        })
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # On Colab, we run this in a thread or background
    app.run(host='0.0.0.0', port=5000)
