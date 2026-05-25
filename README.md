# Hospital Readmission Prediction · ACIF104

Sistema completo para predicción de readmisión hospitalaria en pacientes diabéticos, con backend (API REST), frontend (interfaz web), explicabilidad mediante SHAP y monitoreo del desempeño.

**Asignatura:** ACIF104 — Aprendizaje de Máquina · UNAB 2026
**Equipo:** Equipo 3

## Servicio en Producción

El sistema está disponible en producción en:

**https://hospital-readmission-ml.axiomia.ai/**

| Endpoint | URL |
|---|---|
| **Interfaz web** | https://hospital-readmission-ml.axiomia.ai/ |
| **Estado API** | https://hospital-readmission-ml.axiomia.ai/api/ |
| **Health check** | https://hospital-readmission-ml.axiomia.ai/health |
| **Info del modelo** | https://hospital-readmission-ml.axiomia.ai/api/model-info |
| **Predicción** | https://hospital-readmission-ml.axiomia.ai/api/predict |
| **Monitoreo** | https://hospital-readmission-ml.axiomia.ai/api/monitor |
| **Docs interactivos** | https://hospital-readmission-ml.axiomia.ai/docs |

## Modelo del Sistema

El sistema utiliza un **ensemble ponderado calibrado** que combina tres clasificadores complementarios basados en árboles:

- **Random Forest** — explicabilidad exacta mediante Tree SHAP
- **XGBoost** — captura interacciones no lineales con regularización L1/L2
- **LightGBM** — diversidad metodológica con crecimiento por hojas

Las probabilidades del ensemble se procesan mediante **calibración isotónica** ajustada en validación, garantizando interpretación clínica directa.

### Sistema clínico de tres niveles de riesgo

| Nivel | Rango de probabilidad | Protocolo clínico |
|-------|-----------------------|-------------------|
| 🟢 BAJO | `[0 — 0,35)` | Seguimiento ambulatorio estándar; educación en autocuidado |
| 🟠 MODERADO | `[0,35 — 0,55)` | Seguimiento telefónico a 7 días; revisión farmacológica |
| 🔴 ALTO | `[0,55 — 1,00]` | Intervención preventiva inmediata; visita domiciliaria a 48 h |

**Umbral clínico binario:** `0,42` (Recall ≥ 0,85 sobre la clase positiva).

## Estructura del Repositorio

```
hospital-readmission-ml/
│
├── README.md                          # Este archivo
├── requirements.txt                   # Dependencias Python con versiones
├── .gitignore                         # Exclusiones del control de versiones
│
├── data/                              # Datasets (se descarga aparte)
│   ├── README.md                      # Cómo obtener hospital_readmissions.csv
│   └── hospital_readmissions.csv      # Archivo hospital_readmissions.csv
│
├── notebooks/
│   └── readmision_hospitalaria_colab.ipynb   # Pipeline completo (16 pasos)
│
├── backend/
│   ├── main.py                        # API REST con FastAPI
│   └── model/                        # Generado al ejecutar el notebook
│       ├── rf_final.pkl		# Random Forest optimizado (también usado para Tree SHAP)
│       ├── xgb_final.pkl		# XGBoost
│       ├── lgb_final.pkl		# LightGBM
│       ├── isotonic_calibrator.pkl	# Calibrador isotónico
│       ├── scaler.pkl			# StandardScaler ajustado sobre los 21 features
│       └── ensemble_config.json		# Pesos del ensemble y umbrales del sistema
│
├── frontend/                          # Componente separado y documentado
│   ├── index.html                     # Interfaz web
│   ├── style.css                      # Hoja de estilos
│   └── app.js                         # Lógica del cliente
│
├── figures/                           # Visualizaciones generadas
└── logs/                              # Log de predicciones (runtime)

```

# MANUAL DE INSTALACIÓN Y DESPLIEGUE

**Sistema de Predicción de Readmisión Hospitalaria**

**ACIF104 — Aprendizaje de Máquina**

UNIVERSIDAD ANDRÉS BELLO · Facultad de Ingeniería · UNAB Online

Equipo 3 · NRC 2182 · Santiago, Chile — 2026

---

## Índice

1. [Introducción](#1-introducción)
2. [Requisitos del Sistema](#2-requisitos-del-sistema)
3. [Instalación del Entorno](#3-instalación-del-entorno)
4. [Generación del Modelo](#4-generación-del-modelo)
5. [Despliegue Local Sin Docker](#5-despliegue-local-sin-docker)
6. [Despliegue con Docker](#6-despliegue-con-docker)
7. [Despliegue en Producción](#7-despliegue-en-producción)
8. [Verificación del Despliegue](#8-verificación-del-despliegue)
9. [Operación y Monitoreo](#9-operación-y-monitoreo)
10. [Solución de Problemas](#10-solución-de-problemas)
11. [Anexo. Comandos de Referencia Rápida](#anexo-comandos-de-referencia-rápida)

---

## 1. Introducción

Este manual describe el procedimiento completo para instalar, ejecutar y desplegar como servicio el sistema de Predicción de Readmisión Hospitalaria desarrollado en el marco del proyecto del curso ACIF104 — Aprendizaje de Máquina (UNAB 2026, Equipo 3).

El sistema está compuesto por los siguientes componentes:

- **Notebook Jupyter:** pipeline completo de entrenamiento del modelo en 16 pasos secuenciales.
- **Backend FastAPI:** servicio REST unificado que expone el modelo entrenado mediante cinco endpoints y sirve también el frontend estático.
- **Frontend Web:** interfaz HTML/CSS/JavaScript para uso clínico interactivo, servida directamente por FastAPI.

El servicio unificado (backend + frontend) está disponible en producción en **https://hospital-readmission-ml.axiomia.ai/**. El manual describe también las modalidades de ejecución local con Python y contenerizada con Docker.

---

## 2. Requisitos del Sistema

### 2.1 Hardware

| Componente | Mínimo | Recomendado |
|---|---|---|
| **CPU** | 2 núcleos | 4 núcleos o más |
| **Memoria RAM** | 4 GB | 8 GB o más |
| **Disco** | 5 GB libres | 10 GB libres |
| **Red** | Conexión a internet (instalación) | Conexión continua |

### 2.2 Software base

| Requisito | Versión mínima | Notas |
|---|---|---|
| **Sistema operativo** | Linux / macOS / Windows 10+ | *WSL2 recomendado para Windows* |
| **Python** | 3.11 | *Necesario para todas las dependencias* |
| **pip** | 23.0 | *Gestor de paquetes Python* |
| **Git** | 2.30+ | *Para clonar el repositorio* |
| **Navegador web** | Chrome 100+ / Firefox 100+ | *Para acceder al frontend* |

### 2.3 Software opcional (Docker)

Para el despliegue contenerizado se requieren las siguientes versiones mínimas:

- Docker Engine 24.0 o superior
- Docker Compose 2.20 o superior

### 2.4 Dependencias Python

El repositorio incluye dos archivos de dependencias según el contexto de uso:

#### `requirements.txt` — Desarrollo y entrenamiento completo

Incluye todas las bibliotecas necesarias para ejecutar el notebook de entrenamiento y el backend:

```
pandas==2.0.3                # Manipulación de datos
numpy==1.26.4                # Cálculo numérico
scikit-learn==1.4.2          # Modelos ML clásicos
imbalanced-learn==0.11.0     # SMOTE y otras técnicas de balanceo
xgboost==2.0.3               # Componente del ensemble
lightgbm==4.3.0              # Componente del ensemble
torch==2.1.0                 # Red neuronal MLP
shap==0.44.1                 # Explicabilidad del modelo
fastapi==0.110.0             # Framework del backend
uvicorn==0.27.1              # Servidor ASGI para FastAPI
matplotlib==3.8.3            # Visualizaciones
seaborn==0.13.2              # Visualizaciones estadísticas
jupyter==1.0.0               # Notebook interactivo
joblib==1.3.2                # Serialización de modelos
pydantic==2.6.4              # Validación de esquemas
```

#### `requirements-railway.txt` — Producción (backend únicamente)

Versión reducida para el despliegue en producción (Railway / Docker). Excluye las dependencias exclusivas del entrenamiento (`torch`, `jupyter`, `matplotlib`, `seaborn`, `imbalanced-learn`), reduciendo el tamaño de la imagen en aproximadamente 800 MB:

```
pandas==2.0.3
numpy==1.26.4
scikit-learn==1.4.2
xgboost==2.0.3
lightgbm==4.3.0
shap==0.44.1
fastapi==0.110.0
uvicorn==0.27.1
joblib==1.3.2
pydantic==2.6.4
```

---

## 3. Instalación del Entorno

### 3.1 Clonar el repositorio

```bash
git clone https://github.com/axiomia/hospital-readmission-ml.git
cd hospital-readmission-ml
```

### 3.2 Crear el entorno virtual de Python

**Linux y macOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3.3 Instalar las dependencias

```bash
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota:** la primera instalación puede tardar entre 5 y 15 minutos debido al tamaño de PyTorch y XGBoost.

### 3.4 Verificar la instalación

```bash
python -c "import torch, sklearn, xgboost, lightgbm, shap, fastapi; print('OK')"
```

Si la salida es `OK`, la instalación está completa.

---

## 4. Generación del Modelo

Antes de ejecutar el backend es necesario entrenar el modelo y generar los seis artefactos del ensemble.

### 4.1 Obtener el dataset

Descarga el archivo `hospital_readmissions.csv` desde alguna de las siguientes fuentes:

- **UCI Repository:** https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
- **Kaggle:** https://www.kaggle.com/datasets/dubradave/hospital-readmissions

Coloca el archivo en la carpeta `data/` del proyecto:

```
data/hospital_readmissions.csv
```

### 4.2 Ejecutar el notebook

```bash
jupyter notebook notebooks/readmision_hospitalaria_colab.ipynb
```

El notebook debe ejecutarse en orden secuencial mediante el menú **Cell → Run All**. El proceso completo tarda entre 5 y 15 minutos según el hardware disponible.

### 4.3 Verificar artefactos generados

Al finalizar el paso 16 deben existir seis archivos en el directorio `backend/model/`:

| Archivo | Descripción |
|---|---|
| `rf_final.pkl` | Random Forest entrenado (también se usa para Tree SHAP) |
| `xgb_final.pkl` | XGBoost entrenado |
| `lgb_final.pkl` | LightGBM entrenado |
| `isotonic_calibrator.pkl` | Calibrador isotónico ajustado en validación |
| `scaler.pkl` | StandardScaler ajustado sobre los 21 features |
| `ensemble_config.json` | Pesos del ensemble y umbrales clínicos |

---

## 5. Despliegue Local Sin Docker

### 5.1 Opción manual — Backend

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Si el inicio es exitoso, en consola se mostrará:

```
INFO: Uvicorn running on http://0.0.0.0:8000
INFO: ✓ Ensemble cargado correctamente
INFO:   • Random Forest   (peso: 0.6518)
INFO:   • XGBoost         (peso: 0.6442)
INFO:   • LightGBM        (peso: 0.6408)
INFO:   • Calibrador isotónico ajustado
INFO:   • Tree SHAP listo sobre el componente Random Forest
```

### 5.2 Opción manual — Frontend

En un terminal separado, ejecutar:

```bash
cd frontend
python -m http.server 3000
```

Luego abrir el navegador en: http://localhost:3000

---

## 6. Despliegue con Docker

El proyecto incluye los archivos necesarios para construir las imágenes y levantar el stack completo con Docker Compose.

### 6.1 Construir y levantar el stack

```bash
cd docker
docker compose up -d --build
```

Esto crea dos imágenes y levanta dos contenedores:

| Servicio | Imagen | Puerto | Tecnología |
|---|---|---|---|
| **backend** | `readmission-backend:1.0.0` | 8000 | FastAPI + uvicorn (2 workers) |
| **frontend** | `readmission-frontend:1.0.0` | 3000 | Nginx 1.25 sobre Alpine |

### 6.2 Comandos útiles

```bash
docker compose logs -f          # Ver logs en tiempo real
docker compose ps               # Estado de los servicios
docker compose restart backend  # Reiniciar el backend
docker compose down             # Detener y eliminar contenedores
docker compose down --rmi all   # Detener y borrar imágenes
```

### 6.3 Persistencia de datos

El `docker-compose.yml` monta dos volúmenes para mantener la información entre reinicios:

- **`backend/model/` → `/app/backend/model` (solo lectura):** persistencia de los modelos entrenados.
- **`logs/` → `/app/logs` (lectura/escritura):** acumulación del log de predicciones para análisis posterior.

---

## 7. Despliegue en Producción

El servicio está desplegado y disponible públicamente en **https://hospital-readmission-ml.axiomia.ai/**. El stack de producción utiliza un único servicio FastAPI (backend + frontend estático) desplegado en Railway con `requirements-railway.txt` como conjunto de dependencias.

### 7.1 Variables de entorno

El backend admite las siguientes variables para personalizar su comportamiento:

| Variable | Valor por defecto | Descripción |
|---|---|---|
| `MODEL_DIR` | `/app/backend/model` | Ubicación de los archivos `.pkl` del ensemble. |
| `LOG_PATH` | `/app/logs/predictions.log` | Archivo del log estructurado de predicciones. |
| `CORS_ORIGINS` | `*` | Orígenes permitidos por CORS, separados por coma. |

### 7.2 Recomendaciones de producción

| Aspecto | Recomendación |
|---|---|
| **HTTPS** | Obligatorio en producción. Usar Let's Encrypt o certificado institucional. |
| **CORS** | Restringir `CORS_ORIGINS` a dominios específicos (no usar el comodín `*`). |
| **Workers uvicorn** | Usar 2× núcleos CPU disponibles, mínimo 2. |
| **Logging** | Centralizar logs con ELK, Grafana Loki o servicio similar. |
| **Backup** | Respaldar diariamente `backend/model/` y `logs/`. |
| **Reentrenamiento** | Programar mediante cron o Airflow al menos mensualmente. |
| **Monitoreo** | Prometheus + Grafana sobre los endpoints `/health` y `/monitor`. |
| **Seguridad** | Restringir el acceso a la API mediante token JWT o API key en producción real. |

### 7.3 Despliegue tras un proxy reverso

Configuración recomendada para Nginx con HTTPS:

```nginx
server {
    listen 443 ssl http2;
    server_name predict.hospital.example.com;

    ssl_certificate     /etc/letsencrypt/live/.../fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/.../privkey.pem;

    location / {
        proxy_pass http://localhost:3000/;
        proxy_set_header Host $host;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## 8. Verificación del Despliegue

### 8.1 Health check del backend

```bash
curl http://localhost:8000/health
```

Salida esperada (HTTP 200) en formato JSON:

```json
{
  "status": "ok",
  "model_loaded": true,
  "components": {
    "random_forest":  true,
    "xgboost":        true,
    "lightgbm":       true,
    "calibrator":     true,
    "scaler":         true,
    "shap_explainer": true
  },
  "timestamp": "2026-04-30T15:23:42"
}
```

### 8.2 Predicción de prueba

```bash
curl -X POST http://localhost:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{ "time_in_hospital": 5, "n_lab_procedures": 45,
        "n_procedures": 1, "n_medications": 18,
        "n_outpatient": 0, "n_inpatient": 2,
        "n_emergency": 0, "age_enc": 4,
        "glucose_test_enc": 0, "A1Ctest_enc": 2,
        "change_enc": 1, "diabetes_med_enc": 1,
        "medical_specialty_enc": 3, "diag_1_enc": 2,
        "diag_2_enc": 1, "diag_3_enc": 4 }'
```

### 8.3 Verificación visual del frontend

**Servicio en producción:** abrir https://hospital-readmission-ml.axiomia.ai/ directamente en el navegador.

**Entorno local:** abrir http://localhost:8000 (servicio único FastAPI) y verificar que:

- El formulario carga correctamente con valores por defecto.
- Al pulsar «Predecir riesgo de readmisión» se muestra el indicador semafórico (verde, naranja o rojo).
- El gráfico SHAP muestra las cinco variables más influyentes en la predicción.
- El panel «Monitoreo del Modelo» carga estadísticas tras pulsar «Actualizar».

---

## 9. Operación y Monitoreo

### 9.1 Endpoints disponibles

| Método | Ruta | Propósito | Cuándo usarlo |
|---|---|---|---|
| **GET** | `/` | Frontend estático | Sirve la interfaz web (index.html). |
| **GET** | `/health` | Diagnóstico profundo (alias) | Sondas de Railway / Kubernetes (livenessProbe / readinessProbe). |
| **GET** | `/api/` | Estado general del servicio | Health check ligero, comprueba que el servicio responde. |
| **GET** | `/api/health` | Diagnóstico profundo | Estado de cada componente del ensemble. |
| **GET** | `/api/model-info` | Configuración del modelo | Auditoría de pesos y umbrales en producción. |
| **POST** | `/api/predict` | Predicción individual | Núcleo del sistema. Recibe 16 atributos clínicos. |
| **GET** | `/api/monitor` | Estadísticas agregadas | Detección de deriva conceptual del modelo. |
| **GET** | `/docs` | Documentación interactiva | Swagger UI generado automáticamente por FastAPI. |

### 9.2 Detección de deriva conceptual

El sistema implementa un mecanismo automático para detectar deriva conceptual del modelo. La distribución esperada del entrenamiento es BAJO 21 %, MODERADO 46 % y ALTO 33 %. Cuando la distribución observada se desvía más de 10 puntos porcentuales en cualquier nivel, el frontend muestra una alerta visual y se recomienda investigar la causa raíz.

### 9.3 Auditoría de predicciones

Todas las predicciones se registran en el archivo `logs/predictions.log` en formato JSON Lines, con timestamp ISO 8601, las 16 variables de entrada, la probabilidad calibrada, el nivel de riesgo y la clasificación binaria. Esto permite la trazabilidad completa de cada decisión asistida por el sistema.

### 9.4 Reentrenamiento del modelo

Cuando se detecte deriva conceptual o se acumulen suficientes predicciones nuevas:

1. Ejecutar el notebook con los datos actualizados.
2. Reemplazar los archivos `.pkl` en `backend/model/`.
3. Reiniciar el backend mediante `docker compose restart backend`.

---

## 10. Solución de Problemas

### 10.1 El backend no arranca

**Síntoma:** `FileNotFoundError: rf_final.pkl` al iniciar uvicorn.

**Causa:** los artefactos del modelo no fueron generados.

**Solución:** ejecutar el notebook hasta el paso 16 o copiar los archivos `.pkl` desde otro entorno donde sí existan.

### 10.2 El frontend no se conecta al backend

**Síntoma:** al pulsar «Predecir» aparece error de red o respuesta vacía.

**Causa:** CORS bloqueado, backend no accesible o URL incorrecta.

**Solución 1:** verificar que el backend responde con `curl http://localhost:8000/health`.

**Solución 2:** sobreescribir la URL del API en la query string del navegador, por ejemplo: `http://localhost:8000/?api=http://otro-host:8000`

### 10.3 Error de memoria al cargar XGBoost o LightGBM

**Causa:** RAM insuficiente para cargar el ensemble en memoria.

**Solución:** asegurar al menos 4 GB libres antes de iniciar el backend, o aumentar el límite de memoria del contenedor en `docker-compose.yml`.

### 10.4 Logs llenando el disco

**Solución:** configurar rotación de logs con logrotate (Linux). Ejemplo de configuración en `/etc/logrotate.d/readmission`:

```
/path/to/logs/predictions.log {
    daily
    rotate 30
    compress
    missingok
    notifempty
    copytruncate
}
```

---

## Anexo. Comandos de Referencia Rápida

### Ciclo de vida

```bash
docker compose up -d              # Arranque con Docker
docker compose logs -f            # Ver logs en tiempo real
docker compose ps                 # Estado de los servicios
docker compose down               # Detener y eliminar contenedores
```

### Diagnóstico

```bash
curl http://localhost:8000/health          # Health check (alias raíz)
curl http://localhost:8000/api/            # Estado general de la API
curl http://localhost:8000/api/health      # Diagnóstico profundo del ensemble
curl http://localhost:8000/api/model-info  # Configuración del modelo
curl http://localhost:8000/api/monitor     # Estadísticas agregadas
```

### Desarrollo

```bash
source venv/bin/activate          # Activar entorno virtual
jupyter notebook                  # Ejecutar el notebook
uvicorn main:app --reload         # Backend en modo desarrollo
python3 -m http.server 3000       # Frontend estático local
```

---

*Equipo 3 — ACIF104 Aprendizaje de Máquina · UNAB 2026*

- **Servicio en producción:** https://hospital-readmission-ml.axiomia.ai/
- **Repositorio:** https://github.com/axiomia/hospital-readmission-ml

## Equipo

Equipo 3 — ACIF104 Aprendizaje de Máquina — UNAB 2026

- CRISTÓBAL ACEVEDO MUÑOZ
- JUAN RAMIREZ GOMEZ
- FELIPE TAPIA ARMIJO
- MARCO URRUTIA MOLINA

## Licencia

MIT — uso académico libre para fines educativos.
