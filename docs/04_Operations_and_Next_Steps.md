# 04 Operations and Next Steps

## 1. Local Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

Run application:

```bash
streamlit run app.py
```

## 2. Runtime Expectations
- First startup may take longer because CLIP model weights are loaded.
- Inference is CPU-based; latency depends on hardware.
- Similarity search is fast due to precomputed embeddings.

## 3. Troubleshooting
- Error: database missing
  - Verify `embeddings.npy` and `ids.npy` exist at project root.

- Missing recommendation images
  - Verify image folder structure and article ID naming.

- Metadata lookup issues
  - Ensure `article_id` is loaded as string and matches IDs exactly.

## 4. Suggested Enhancements
1. Add text-to-image and text-to-catalog search using CLIP text encoder.
2. Move vectors to a dedicated vector database for large scale.
3. Add hybrid ranking with behavioral signals.
4. Add offline evaluation metrics for retrieval quality.
5. Add CI checks for artifact consistency.
