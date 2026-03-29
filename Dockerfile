FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spacy model (needed for PII detection)
RUN python -m spacy download en_core_web_sm || true

# App code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Create data dirs
RUN mkdir -p /app/data /app/uploads /app/logs

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]