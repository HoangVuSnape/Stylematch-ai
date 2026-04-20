FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements_local.txt .
RUN pip install --no-cache-dir -r requirements_local.txt

# Copy application code
COPY app_local.py .

# Environment variables
ENV COLAB_URL=https://f79f-34-147-69-24.ngrok-free.app
ENV PORT=8000

EXPOSE 8000

CMD ["python", "app_local.py"]
