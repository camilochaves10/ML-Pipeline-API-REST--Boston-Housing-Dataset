# Predicción de Precios de Viviendas con MLOps

Proyecto de Machine Learning y MLOps para predicción de precios de viviendas utilizando el Boston Housing Dataset. El sistema incluye entrenamiento automático de modelos, tracking de experimentos con MLflow, una API con FastAPI y un dashboard interactivo con Streamlit.

## Tecnologías

- Python
- Scikit-learn
- FastAPI
- Streamlit
- MLflow
- Docker & Docker Compose
- GitHub Actions

---

## Modelos Evaluados

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

## Estructura del Proyecto

```bash
.
├── app/
│   ├── api.py
│   └── streamlit_app.py
│
├── src/
│   ├── pipeline.py
│   └── train.py
│
├── data/
│   └── raw/
│       ├── HousingData.csv
│       └── sample.csv
│
├── models/
│   ├── housing_model.joblib
│   ├── metrics.json
│   └── feature_importance.json
│
├── mlflow_data/
├── logs/
├── tests/
│   └── test_api.py
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.streamlit
├── Makefile
├── requirements.txt
└── README.md
```

---

## Para Ejecutar el Proyecto:

Clonar el repositorio:

```bash
git clone <repo-url>
cd <repo-name>
```

Ejecutar todo el pipeline:

```bash
make run-docker-compose
```

### Streamlit

http://localhost:8501

### FastAPI Docs

http://localhost:8000/docs

### MLflow

http://localhost:5050

---

## Funcionalidades

- Entrenamiento automático de múltiples modelos.
- Selección automática del mejor modelo.
- Tracking de experimentos con MLflow.
- API REST para inferencia.
- Dashboard interactivo con Streamlit.
- Arquitectura completamente dockerizada.

---

## Métricas Evaluadas

- MAE
- RMSE
- R² Score

---

## Comandos Útiles

Detener servicios:

```bash
make stop-docker-compose
```

Ver logs:

```bash
make logs
```
