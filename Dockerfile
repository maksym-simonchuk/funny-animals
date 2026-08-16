FROM python:3.12-slim

# ffmpeg — нужен processors/video.py (ffprobe/транскод); build tools для колёс torch/opencv не нужны на slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Слой зависимостей отдельно от кода — кэшируется, пока requirements.txt не меняется
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# data/ (видео, кадры, логи, модели) — volume, монтируется в docker-compose.yml
RUN mkdir -p data/videos data/frames data/thumbnails data/logs data/models \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "app.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
