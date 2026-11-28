FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY deploy/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy/app.py deploy/check.py deploy/model.tflite ./
RUN python check.py

# Streamlit necesita HTTP + WebSocket; Vercel inyecta el puerto en $PORT.
# CORS/XSRF se desactivan porque el proxy de Vercel cambia el host original.
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-80} --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]
