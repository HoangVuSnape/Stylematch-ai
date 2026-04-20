# 01 Repository Contents

## 1. Top-Level Files
- `app.py`
  - Main Streamlit application.
  - Loads model and precomputed vectors.
  - Handles image upload and recommendation display.

- `articles.csv`
  - Product metadata table.
  - Must contain `article_id` and descriptive fields (for example `prod_name`).

- `embeddings.npy`
  - Precomputed embedding matrix for catalog items.
  - Used at runtime for fast similarity scoring.

- `ids.npy`
  - Item ID array aligned by index with `embeddings.npy`.

- `data_processing_and_embedding.ipynb`
  - Offline notebook to prepare and export embeddings.

- `requirements.txt`
  - Python dependency list.

- `README.md`
  - Public project description and quickstart.

## 2. Main Directories
- `images/`
  - Catalog image store.
  - Organized by article ID prefix subfolders (`012/`, `014/`, etc.).

- `docs/`
  - Internal technical documentation (this folder).

## 3. Data Contract Between Offline and Online Phases
Required alignment:
- Row/index in `embeddings.npy` must match index in `ids.npy`.
- Values in `ids.npy` must map to `article_id` in `articles.csv`.
- Image path resolution must find corresponding JPG files under `images/`.

If this contract breaks, recommendations may be incorrect or missing.
