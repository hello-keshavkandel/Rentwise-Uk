# RentWise UK — portable container image for the Streamlit dashboard.
# Build:  docker build -t rentwise-uk .
# Run:    docker run -p 8501:8501 rentwise-uk   ->  http://localhost:8501
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code + the committed artifacts the app serves (model, processed data, conformal)
COPY . .

EXPOSE 8501

# Container health = the app's own health check (artifacts present & serviceable)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -m src.health || exit 1

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
