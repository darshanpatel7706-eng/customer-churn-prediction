# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Requirements copy karo
COPY requirements.txt .

# Install karo
RUN pip install --no-cache-dir -r requirements.txt

# Pura project copy karo
COPY . .

# Port open karo
EXPOSE 8000

# App run karo
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]