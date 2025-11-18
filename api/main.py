#!/usr/bin/env python
# coding: utf-8

import os
import json
import pickle
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from models.ncf_model import NCF

# ---------------------------------------------------------------------------
# Environment and threading limits (keeps BLAS/FAISS from over-threading)
# ---------------------------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
faiss.omp_set_num_threads(1)
torch.set_grad_enabled(False)

# ---------------------------------------------------------------------------
# Paths / configuration
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]  # project root (MOVIELENS_DL/)
ART_DIR = ROOT / "features_artifacts"
CKPT_DIR = ROOT / "checkpoints"
MAPPINGS_DIR = ROOT / "mappings"

# Embeddings + IDs
EMB_NPZ = ART_DIR / "movie_embeddings.npz"
# EMB_NPZ = ART_DIR / "movie_embeddings_finetuned.npz" # for finetuned ones instead

FAISS_IVF = ART_DIR / "movie_faiss_index_flat.index"
NCF_CKPT = CKPT_DIR / "ncf_finetuned_final.pt"
U2I_JSON = MAPPINGS_DIR / "u2i.json"              # {userId: u_idx}
SEEN_PKL = MAPPINGS_DIR / "user_seen_items.pkl"   # {u_idx -> set(m_idx)}
ID2TITLE_JSON = MAPPINGS_DIR / "id_to_title.json" # {tmdb_id: title}

# Hybrid tuning defaults (from Step 7D)
LOW_HISTORY_MAX_SEEN = 20
C_FOR_LOW_USERS = 3.0
TOPK_DEFAULT = 20
SHORTLIST_M = 200

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _safe_l2_normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32, order="C")
    np.nan_to_num(mat, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms

def _safe_l2_normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(1, -1)
    np.nan_to_num(v, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return (v / n).astype(np.float32)

# ---------------------------------------------------------------------------
# Load embeddings and IDs
# ---------------------------------------------------------------------------
npz = np.load(str(EMB_NPZ), allow_pickle=False)

# Accept either key naming convention
if "emb" in npz and "tmdb_id" in npz:
    emb = npz["emb"]
    ids = npz["tmdb_id"]
elif "movie_embeddings" in npz and "tmdb_ids" in npz:
    emb = npz["movie_embeddings"]
    ids = npz["tmdb_ids"]
else:
    raise ValueError(f"Unrecognized keys in {EMB_NPZ}: {list(npz.keys())}")

movie_embeddings = _safe_l2_normalize_rows(np.asarray(emb))
tmdb_ids = np.asarray(ids).astype(np.int64)

id_to_row = {int(t): int(i) for i, t in enumerate(tmdb_ids)}
num_items = movie_embeddings.shape[0]
emb_dim = movie_embeddings.shape[1]

# Optional TMDB id → title map
id_to_title: dict[int, str] = {}
if ID2TITLE_JSON.exists():
    with open(str(ID2TITLE_JSON), "r", encoding="utf-8") as f:
        id_to_title = {int(k): v for k, v in json.load(f).items()}

# ---------------------------------------------------------------------------
# FAISS index (IP over L2-normalized vectors ≈ cosine similarity)
# ---------------------------------------------------------------------------
d = movie_embeddings.shape[1]
if FAISS_IVF.exists():
    try:
        faiss_index = faiss.read_index(str(FAISS_IVF))
        if faiss_index.d != d:
            print(f"[WARN] FAISS index dim {faiss_index.d} != emb dim {d}. Rebuilding in-memory.")
            faiss_index = faiss.IndexFlatIP(d)
            faiss_index.add(movie_embeddings.astype("float32"))
    except Exception as e:
        print(f"[WARN] Failed to read saved FAISS index: {e}. Rebuilding in-memory.")
        faiss_index = faiss.IndexFlatIP(d)
        faiss_index.add(movie_embeddings.astype("float32"))
else:
    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(movie_embeddings.astype("float32"))

print("FAISS ready:", faiss_index.ntotal, "vectors; dim:", d)

# ---------------------------------------------------------------------------
# User mappings / interactions
# ---------------------------------------------------------------------------
u2i: dict[int, int] = {}
if U2I_JSON.exists():
    with open(str(U2I_JSON), "r", encoding="utf-8") as f:
        tmp = json.load(f)
        u2i = {int(k): int(v) for k, v in tmp.items()}

user_seen_items: dict[int, set[int]] = {}
if SEEN_PKL.exists():
    with open(str(SEEN_PKL), "rb") as f:
        raw = pickle.load(f)  # dict[u_idx] -> set(m_idx)
    user_seen_items = {
        int(u): set(int(m) for m in s) for u, s in raw.items()
    }

# ---------------------------------------------------------------------------
# Load NCF checkpoint (content–CF hybrid backbone)
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_device = "cpu"  # always load on CPU first

state = torch.load(str(NCF_CKPT), map_location=ckpt_device)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]

n_users_ckpt = state["user_emb.weight"].shape[0]
n_items_ckpt = state["item_emb.weight"].shape[0]
emb_dim_ckpt = state["item_emb.weight"].shape[1]

model = NCF(
    num_users=n_users_ckpt,
    num_items=n_items_ckpt,
    emb_dim=emb_dim_ckpt,
    init_item_vectors=None,
    freeze_items=False,
).cpu()

missing, unexpected = model.load_state_dict(state, strict=False)
if missing or unexpected:
    print("[INFO] load_state_dict non-strict:", {"missing": missing, "unexpected": unexpected})

model.eval()

# align item embeddings with current content tower
n_items_cur, emb_dim_cur = movie_embeddings.shape
if n_items_cur == n_items_ckpt and emb_dim_cur == emb_dim_ckpt:
    with torch.no_grad():
        model.item_emb.weight.data.copy_(torch.from_numpy(movie_embeddings))

if device == "cuda":
    model = model.to("cuda")

# Guard: mapping consistency
if u2i and max(u2i.values()) >= n_users_ckpt:
    raise RuntimeError(
        f"u2i max index {max(u2i.values())} >= checkpoint user_emb size {n_users_ckpt}. "
        "Use the same split mapping you trained with or retrain."
    )

# ---------------------------------------------------------------------------
# helpers: content profile, shortlist, NCF scoring, hybrid scoring
# ---------------------------------------------------------------------------
def build_user_profile_mean(u_idx: int) -> Optional[np.ndarray]:
    seen = list(user_seen_items.get(u_idx, set()))
    if not seen:
        return None
    prof = movie_embeddings[seen].mean(axis=0)
    n = np.linalg.norm(prof)
    if n == 0:
        return None
    return (prof / n).astype(np.float32)

def shortlist_by_content_seed(seed_tmdb_id: int, M: int = SHORTLIST_M) -> np.ndarray:
    pos = id_to_row.get(int(seed_tmdb_id))
    if pos is None:
        raise KeyError(f"TMDB id {seed_tmdb_id} not found.")
    q = _safe_l2_normalize_vec(movie_embeddings[pos])
    sims, idxs = faiss_index.search(q, M)
    idxs = idxs[0]
    idxs = idxs[idxs != pos]  # drop self
    return idxs.astype(np.int32, copy=False)

def shortlist_by_content_for_user(u_idx: int, M: int = SHORTLIST_M) -> np.ndarray:
    prof = build_user_profile_mean(u_idx)
    seen = user_seen_items.get(u_idx, set())
    if prof is None:
        # fallback: first unseen M
        if len(seen) >= num_items:
            return np.array([], dtype=np.int32)
        idxs = [i for i in range(num_items) if i not in seen][:M]
        return np.array(idxs, dtype=np.int32)
    q = _safe_l2_normalize_vec(prof)
    sims, idxs = faiss_index.search(q, M)
    idxs = idxs[0]
    idxs = np.array([i for i in idxs if i not in seen], dtype=np.int32)
    return idxs

@torch.no_grad()
def ncf_score_user_array(u_idx: int, item_idxs: np.ndarray, batch: int = 4096) -> np.ndarray:
    out = []
    u_t = torch.tensor([u_idx], dtype=torch.long, device=device)
    for i in range(0, len(item_idxs), batch):
        chunk = item_idxs[i : i + batch]
        uu = u_t.repeat(len(chunk))
        mm = torch.tensor(chunk, dtype=torch.long, device=device)
        logits = model(uu, mm).detach().cpu().numpy().astype(np.float32)
        out.append(logits)
    return np.concatenate(out, axis=0)

def hybrid_for_user(u_idx: int, item_idxs: np.ndarray, C: float = C_FOR_LOW_USERS):
    # Content: cosine similarity to mean profile
    prof = build_user_profile_mean(u_idx)
    if prof is None:
        c_scores = np.zeros(len(item_idxs), dtype=np.float32)
    else:
        c_scores = (movie_embeddings[item_idxs] @ prof).astype(np.float32)

    # Collaborative: NCF logits over same items
    cf_scores = ncf_score_user_array(u_idx, item_idxs)

    def z(x: np.ndarray) -> np.ndarray:
        m, s = x.mean(), x.std()
        return (x - m) / (s + 1e-9) if s > 0 else x * 0.0

    c2 = z(c_scores)
    cf2 = z(cf_scores)

    # Alpha schedule: more CF weight as user gets more interactions
    n_seen = len(user_seen_items.get(u_idx, set()))
    alpha = min(1.0, C / (C + max(0, n_seen)))
    scores = alpha * c2 + (1.0 - alpha) * cf2
    return scores.astype(np.float32), alpha

def pack_items(item_idxs: np.ndarray, scores: Optional[np.ndarray] = None) -> List[dict]:
    out: List[dict] = []
    for i, idx in enumerate(item_idxs.tolist()):
        tmdb = int(tmdb_ids[idx])
        item = {
            "tmdb_id": tmdb,
            "title": id_to_title.get(tmdb, None),
        }
        if scores is not None:
            item["score"] = float(scores[i])
        out.append(item)
    return out

# ---------------------------------------------------------------------------
# FastAPI app & endpoints
# ---------------------------------------------------------------------------
app = FastAPI(title="Hybrid Recommender API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecResponse(BaseModel):
    items: List[dict]
    alpha: Optional[float] = None
    used_shortlist_M: Optional[int] = None
    k: int

@app.get("/health")
def health():
    return {
        "status": "ok",
        "users": len(u2i),
        "items": int(movie_embeddings.shape[0]),
        "dim": int(movie_embeddings.shape[1]),
        "sample_user_ids": list(list(u2i.keys())[:5]),
    }

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.get("/recommend_by_movie", response_model=RecResponse)
def recommend_by_movie(
    tmdb_id: int,
    k: int = Query(TOPK_DEFAULT, ge=1, le=100),
    M: int = SHORTLIST_M,
):
    try:
        idxs = shortlist_by_content_seed(tmdb_id, M=M)[:k]
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return RecResponse(items=pack_items(idxs), alpha=1.0, used_shortlist_M=M, k=k)

@app.get("/recommend_for_user", response_model=RecResponse)
def recommend_for_user(
    user_id: int,
    k: int = Query(TOPK_DEFAULT, ge=1, le=100),
    M: int = SHORTLIST_M,
):
    if user_id not in u2i:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")
    u_idx = u2i[user_id]
    shortlist = shortlist_by_content_for_user(u_idx, M=M)
    if shortlist.size == 0:
        return RecResponse(items=[], alpha=None, used_shortlist_M=M, k=k)

    scores = ncf_score_user_array(u_idx, shortlist)
    order = np.argsort(-scores)[:k]
    topk = shortlist[order]
    return RecResponse(items=pack_items(topk, scores[order]), alpha=None, used_shortlist_M=M, k=k)

@app.get("/recommend_hybrid", response_model=RecResponse)
def recommend_hybrid(
    user_id: int,
    tmdb_id: Optional[int] = None,
    k: int = Query(TOPK_DEFAULT, ge=1, le=100),
    M: int = SHORTLIST_M,
    C: float = C_FOR_LOW_USERS,
):
    # Movie→movie (content only) path
    if tmdb_id is not None:
        idxs = shortlist_by_content_seed(tmdb_id, M=M)[:k]
        return RecResponse(items=pack_items(idxs), alpha=1.0, used_shortlist_M=M, k=k)

    # User→movies (hybrid) path
    if user_id not in u2i:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found")

    u_idx = u2i[user_id]
    shortlist = shortlist_by_content_for_user(u_idx, M=M)
    if shortlist.size == 0:
        return RecResponse(items=[], alpha=None, used_shortlist_M=M, k=k)

    scores, alpha = hybrid_for_user(u_idx, shortlist, C=C)
    order = np.argsort(-scores)[:k]
    topk = shortlist[order]
    return RecResponse(
        items=pack_items(topk, scores[order]),
        alpha=float(alpha),
        used_shortlist_M=M,
        k=k,
    )
