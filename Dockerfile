# PhishGuard API + web UI.
#   docker build -t phishguard .                                   # full ensemble (classic + DistilBERT, CPU)
#   docker build -t phishguard-lite --build-arg MODEL=classic .    # small image, classic model only
# Library versions are pinned to the ones used for training: pickled scikit-learn models must match exactly.
FROM python:3.14-slim

ARG MODEL=ensemble
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PHISHGUARD_MODEL=${MODEL}

RUN pip install --no-cache-dir scikit-learn==1.9.1 scipy==1.18.1 numpy==2.5.4 pandas==3.1.1 joblib==1.6.0 \
        xgboost==3.4.1 beautifulsoup4==4.15.0 tldextract==5.4.0 fastapi==0.162.7 "uvicorn[standard]" \
        python-multipart pydantic requests google-api-python-client google-auth-oauthlib msal \
 && if [ "$MODEL" != "classic" ]; then \
        pip install --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu \
     && pip install --no-cache-dir transformers==5.23.0; fi

COPY phishguard/ phishguard/
COPY models/ models/

EXPOSE 8000
CMD ["uvicorn", "phishguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
