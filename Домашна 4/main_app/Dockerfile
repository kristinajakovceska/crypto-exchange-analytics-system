# main_app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy application code (includes templates + static)
COPY analytics_api ./analytics_api

ENV PORT=8000
EXPOSE 8000

# Use shell so $PORT is expanded
CMD ["sh", "-c", "uvicorn analytics_api.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
