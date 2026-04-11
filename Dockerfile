# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3-slim

WORKDIR /app

# Install system dependencies LightGBM needs
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything including model.txt
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]