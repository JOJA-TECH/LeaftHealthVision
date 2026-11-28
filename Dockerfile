FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/app.py deploy/check.py ./
COPY src/leafthealthvision/evaluation/comparison.py ./comparison.py
COPY deploy/models/ ./models/

# Solo resultados de Colab: métricas, emisiones, artefactos y checkpoints de origen.
COPY experiments/efficientnetb0/colab_results/mlflow.db ./experiments/efficientnetb0/colab_results/
COPY experiments/efficientnetb0/colab_results/emissions.csv ./experiments/efficientnetb0/colab_results/
COPY experiments/efficientnetb0/colab_results/leafthealthvision/artifacts/ ./experiments/efficientnetb0/colab_results/leafthealthvision/artifacts/
COPY experiments/mobilenetv2/colab_results/emissions.csv ./experiments/mobilenetv2/colab_results/
COPY experiments/mobilenetv2/colab_results/leafthealthvision/artifacts/ ./experiments/mobilenetv2/colab_results/leafthealthvision/artifacts/

RUN python check.py

# Streamlit necesita HTTP + WebSocket; Vercel inyecta el puerto en $PORT.
# CORS/XSRF se desactivan porque el proxy de Vercel cambia el host original.
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-80} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
