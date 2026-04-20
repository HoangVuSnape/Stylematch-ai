# 03 Data and Embedding Pipeline

## 1. Input Assets
- Product metadata: `articles.csv`.
- Product images: `images/<prefix>/<article_id>.jpg`.

## 2. Embedding Generation Standard
- Model: `openai/clip-vit-base-patch32`.
- Framework: Hugging Face Transformers + PyTorch.
- Vector dimension: 512.
- Normalization: L2 normalization before similarity search.

## 3. Produced Artifacts
- `embeddings.npy`:
  - Type: float array.
  - Shape: `(N_items, 512)`.

- `ids.npy`:
  - Type: string/object array.
  - Shape: `(N_items,)`.
  - Must align index-wise with embeddings.

## 4. Validation Checklist
Run these checks after regeneration:
1. `len(ids) == embeddings.shape[0]`.
2. `embeddings.shape[1] == 512`.
3. Random sample IDs exist in `articles.csv`.
4. Random sample image paths are resolvable.

## 5. Regeneration Trigger Cases
- New catalog ingestion.
- Model version change.
- Image preprocessing update.
- Data quality fix affecting article IDs.
