FROM python:3.12-slim

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