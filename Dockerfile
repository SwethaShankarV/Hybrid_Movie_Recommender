# Dockerfile
FROM python:3.11-slim

# Avoid FAISS/BLAS thread clashes, keep logs unbuffered
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy code + artifacts (API only)
COPY api ./api
COPY models ./models
COPY features_artifacts ./features_artifacts
COPY checkpoints ./checkpoints
COPY mappings ./mappings

# Import path for `from models...`
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
