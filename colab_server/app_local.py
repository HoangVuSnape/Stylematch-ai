import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from dotenv import load_dotenv # Thêm dòng này
# Load variables from .env file
load_dotenv() # Thêm dòng này
# --- Configuration ---
# This URL will be the ngrok URL from Colab
COLAB_URL = os.getenv("COLAB_URL")
PORT = int(os.getenv("PORT", 8000))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) # Enable CORS for frontend interaction

@app.route('/health', methods=['GET'])
def health():
    # Check if Colab is reachable
    try:
        r = requests.get(f"{COLAB_URL}/health", timeout=5)
        colab_status = r.json()
        return jsonify({
            "local_status": "online",
            "colab_connection": "ok",
            "colab_info": colab_status
        })
    except Exception as e:
        return jsonify({
            "local_status": "online",
            "colab_connection": "failed",
            "error": str(e)
        }), 503

@app.route('/api/search', methods=['POST'])
def proxy_search():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Forward the request to Colab
    try:
        file = request.files['image']
        top_k = request.form.get('top_k', 5)
        
        # We must re-read the file to send it
        files = {'image': (file.filename, file.read(), file.content_type)}
        data = {'top_k': top_k}
        
        logger.info(f"Forwarding search request to Colab: {COLAB_URL}/search")
        response = requests.post(f"{COLAB_URL}/search", files=files, data=data, timeout=30)
        
        return (response.content, response.status_code, response.headers.items())
    
    except Exception as e:
        logger.error(f"Error communicating with Colab: {e}")
        return jsonify({"error": "Gateway error: Could not reach Colab server"}), 502

if __name__ == '__main__':
    logger.info(f"Starting Local Backend on port {PORT}")
    logger.info(f"Target Colab URL: {COLAB_URL}")
    app.run(host='0.0.0.0', port=PORT)
