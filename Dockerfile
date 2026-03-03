FROM python:3.12-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libxcb1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set environment variable for Gemini (will be overridden by Render's env)
ENV GEMINI_API_KEY=${GEMINI_API_KEY}

# Expose port (Render will set the PORT env var)
EXPOSE 10000

# Run the app with gunicorn
CMD gunicorn app.app:app --bind 0.0.0.0:$PORT