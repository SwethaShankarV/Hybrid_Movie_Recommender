# 🎬 Hybrid Movie Recommender System

**Content + Collaborative Neural Hybrid**  
**FAISS + Transformers + NCF | Dockerized FastAPI | Production-Grade Deployment**

---

## Live API  
- **API:** https://hybrid-rec-5th7.onrender.com  
- **Swagger Docs:** https://hybrid-rec-5th7.onrender.com/docs  

---

## Overview

A production-grade hybrid recommender system built using **MovieLens + TMDB** data.

It blends **content-based filtering** and **neural collaborative filtering** to deliver real-time, personalized movie recommendations.

### Components

#### 1. **Content-Based Filtering (CBF)**
- Pretrained text embeddings for metadata  
- L2-normalized feature vectors  
- FAISS `IndexFlatIP` for fast similarity search  
- Excellent for cold-start scenarios  

#### 2. **Collaborative Filtering (CF)**
- Neural Collaborative Filtering (NCF)
- Learns user–item interactions  
- Dense embedding towers for users & items  
- Trained on MovieLens ratings  

#### 3. **Hybrid Scoring**
Uses an adaptive α-blend: score = α · content + (1 − α) · nc_cf

- Cold users → higher content weighting  
- Active users → higher CF weighting  

Produces stable and personalized results.

---

## Project Architecture
```
MOVIELENS_DL/
│
├── api/                     # FastAPI backend
│   └── main.py              # Core API logic (clean production version)
│
├── models/
│   └── ncf_model.py         # Neural Collaborative Filtering model
│
├── features_artifacts/      # Precomputed embeddings + FAISS index
│
├── checkpoints/             # Trained NCF model weights
│
├── mappings/
│   ├── u2i.json             # UserID → Index map
│   ├── user_seen_items.pkl  # User → watched movies
│   └── id_to_title.json     # TMDB ID → movie title
│
├── frontend/
│   └── index.html           # Simple frontend for testing
│
├── notebooks/
│   ├── main.ipynb           # Development notebook
│   └── hybrid_rec_training.ipynb
│
├── Dockerfile               # Production image
├── requirements-api.txt     # Minimal API dependencies
└── README.md
```

---

## Key Features

### Hybrid Recommendation Modes
- `/recommend_by_movie` — Similar movies (content only)  
- `/recommend_for_user` — Collaborative filtering  
- `/recommend_hybrid` — Blended scoring  

### FAISS ANN Search
- Fast search over **44k movies × 128-dim vectors**

### NCF Model
- MLP architecture  
- Embedding-based user/item representations  
- Handles cold-start via CBF fallback  

### FastAPI Backend
- Fully typed  
- Clean Pydantic response models  
- Automatic OpenAPI docs  
- CORS enabled  

### Production-Ready Deployment
- Dockerized  
- AMD64-compatible  
- Deployed on Render  
- Health checks enabled  

---

## Live API Examples

### **Health Check**
**GET** `/health`

**Response:**
<code>
{
  "status": "ok",
  "users": 671,
  "items": 44383,
  "dim": 128,
  "sample_user_ids": [1, 2, 3, 4, 5]
}
</code>

# Hybrid Recommendation API

## Endpoints

### Recommend by Movie (content only)
<code>GET /recommend_by_movie?tmdb_id=19995&k=10</code>

### Recommend for User (collaborative only)
<code>GET /recommend_for_user?user_id=42&k=10</code>

### Hybrid Recommendation
<code>GET /recommend_hybrid?user_id=42&k=10</code>

---

## Running Locally with Docker

### 1. Build the image
<code>
docker build -t hybrid-rec
</code>

### 2. Run the container
<code>
docker run -p 8080:8000 hybrid-rec
</code>

### 3. View docs
Open:
http://localhost:8080/docs

## Running Locally (Without Docker)
<pre><code>
pip install -r requirements-api.txt
uvicorn api.main:app --reload
</code></pre>
Open:
http://127.0.0.1:8000/docs

## Deployment (Render)
The production image is hosted on Docker Hub:
<code>
docker.io/swethashankarchowdry/hybrid-rec:amd64
</code>

Deployed on Render as a Web Service with:

- Root path → redirects to `/docs`
- Built-in health checks
- Free tier supported

---

## Tech Stack

### Machine Learning
- PyTorch
- Neural Collaborative Filtering (NCF)
- FAISS (Approximate Nearest Neighbor Search)
- NumPy

### Backend
- FastAPI
- Pydantic
- Uvicorn

### Deployment
- Docker
- Buildx multi-architecture build
- Render Cloud Hosting

### Data
- MovieLens 25M
- TMDB metadata

---

## Future Enhancements
- Add search & movie detail lookup
- Migrate to GPU inference
- Add model re-training pipeline
- Build a React UI for richer interaction
