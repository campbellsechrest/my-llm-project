# --- base ---
FROM python:3.11-slim

# system deps (faster tokenizers, wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# workdir
WORKDIR /app

# copy only requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# copy source
COPY . /app

# uvicorn will bind to 0.0.0.0 in Render
ENV PORT=8000
EXPOSE 8000

# start server
CMD ["uvicorn", "--app-dir", "src", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
