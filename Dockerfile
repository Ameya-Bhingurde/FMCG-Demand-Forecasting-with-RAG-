FROM python:3.10-slim

WORKDIR /app

# System deps Prophet needs for its C++ compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (cached as a layer when only code changes).
# Root requirements.txt delegates to fmcg_genai/requirements.txt, so copy both.
COPY requirements.txt ./requirements.txt
COPY fmcg_genai/requirements.txt ./fmcg_genai/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repo
COPY . .

# HF Spaces convention: bind to 7860 on 0.0.0.0
EXPOSE 7860

# Make start.sh executable (it builds models + vector store on first boot if missing)
RUN chmod +x start.sh || true

# Streamlit needs a writable home for cache
ENV HOME=/tmp \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["bash", "start.sh"]
