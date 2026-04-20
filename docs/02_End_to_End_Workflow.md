# 02 End-to-End Workflow

## 1. Offline Workflow (Build Index)
1. Collect source images and metadata.
2. Run embedding pipeline notebook.
3. Encode each image with CLIP image encoder.
4. L2-normalize vectors.
5. Save `embeddings.npy` and `ids.npy`.
6. Validate shape and index alignment.

Outcome:
- Search-ready vector index files for online serving.

## 2. Online Workflow (Runtime Query)
1. User uploads an image in Streamlit sidebar.
2. App converts image to CLIP input tensor.
3. Model computes query embedding.
4. Query vector is normalized.
5. Dot product with catalog embedding matrix produces similarity scores.
6. Scores are sorted descending, top K indices selected.
7. App resolves image paths and metadata, then renders recommendation cards.

## 3. Workflow Diagram (Text)
`User Upload -> CLIP Query Embedding -> Similarity Search -> Top-K IDs -> Metadata + Images -> UI Results`

## 4. Failure Paths
- Missing `embeddings.npy`: app shows database-missing error.
- Missing image for returned ID: app shows warning for that card.
- ID mismatch between arrays and CSV: metadata lookup may fail.
