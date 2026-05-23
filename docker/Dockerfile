# ═══════════════════════════════════════════════════════════════════
# Dockerfile — Backend FastAPI
# Predicción de Readmisión Hospitalaria
# ACIF104 — Aprendizaje de Máquina · UNAB 2026
# ═══════════════════════════════════════════════════════════════════
FROM python:3.11-slim

# Variables de entorno para Python en contenedor
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Dependencias del sistema (necesarias para xgboost / lightgbm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Crear usuario no privilegiado para mayor seguridad
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /app

# Instalar dependencias Python (cacheable layer)
COPY requirements-railway.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copiar el código del backend y los modelos
COPY backend/ /app/backend/
RUN mkdir -p /app/logs

# Permisos para el usuario no-root
RUN chown -R appuser:appuser /app
USER appuser

# Punto de entrada
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}", "--workers", "1"]
