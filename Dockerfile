# ---- ACR Transcribe API ----
# נבנה עבור Render (שכבה חינמית, 512MB). עובד ללא שינוי גם על
# Google Cloud Run, Fly.io או כל VPS עם Docker.
FROM python:3.12-slim

# ffmpeg + ffprobe — הכלי שממיר AMR/ACR/3GP ל-MP3
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server_api.py .

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/tmp \
    PORT=10000

EXPOSE 10000

# worker אחד — הכי חסכוני בזיכרון. ההמתנה היא על רשת, לא על CPU,
# ולכן worker אחד מטפל בכמה בקשות במקביל בלי בעיה.
CMD ["sh", "-c", "exec uvicorn server_api:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75"]
