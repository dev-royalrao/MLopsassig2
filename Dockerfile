# Dockerfile - expects saved/savedmodel.pth to exist before building (or provided during build)
FROM python:3.10-slim

WORKDIR /app

# Install system deps (Pillow needs libjpeg etc)
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and src & saved model
COPY app ./app
COPY src ./src
# If you have trained model locally, copy it into image:
# (CI pipeline will ensure saved/ is available before building)
COPY saved ./saved

WORKDIR /app/app

ENV FLASK_ENV=production
EXPOSE 5000

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2"]
