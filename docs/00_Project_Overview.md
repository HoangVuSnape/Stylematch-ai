# 00 Project Overview

## 1. What This Project Is
StyleMatch AI is a visual-semantic fashion recommendation system.
It takes an input image and returns fashion items with similar style using vector similarity, instead of keyword-only matching.

## 2. Core Objective
- Solve style-based search where text labels are insufficient.
- Support zero-history recommendations (new items can still be retrieved).
- Keep inference fast enough for an interactive Streamlit app.

## 3. High-Level Architecture
The system has two main phases:

1. Offline indexing phase
- Load fashion images and metadata.
- Encode each image with CLIP into a 512-dimension embedding.
- Normalize and store embeddings in NumPy files.

2. Online query phase
- User uploads an image.
- App encodes query image with the same CLIP model.
- Cosine similarity (dot product on normalized vectors) ranks catalog items.
- Top-K similar items are displayed with metadata.

## 4. Primary Outputs
- `embeddings.npy`: matrix of item embeddings.
- `ids.npy`: array of corresponding article IDs.
- Streamlit UI: query image + top similar recommendations.

## 5. Project Scope
In scope:
- Visual similarity retrieval.
- CPU-friendly deployment path.

Out of scope (current version):
- Personalized user profiles.
- Transaction-history collaborative filtering.
- Production vector database and distributed serving.
