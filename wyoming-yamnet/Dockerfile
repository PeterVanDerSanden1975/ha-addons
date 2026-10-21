FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
      "wyoming>=1.5.0" \
      "numpy>=1.26,<2" \
      "tflite-runtime==2.14.0"

WORKDIR /app
COPY yamnet_server.py /app/yamnet_server.py

# Model + klassenlijst worden bij de eerste start naar /data gehaald (persistent).
CMD ["python3", "-u", "/app/yamnet_server.py"]
