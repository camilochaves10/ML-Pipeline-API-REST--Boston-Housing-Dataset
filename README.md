# Housing Price Prediction MLOps MVP

This project implements an end-to-end machine learning pipeline using the Boston Housing open-source dataset.

It includes:

- Dataset cleaning and preprocessing
- Reproducible sklearn pipeline
- Regression model training
- Model evaluation
- Model persistence
- REST API with FastAPI
- Streamlit frontend
- Basic production monitoring
- Retraining support
- Docker support
- GitHub Actions CI
- Optional Airflow orchestration

---

## Project Structure

```text
housing-mlops/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── src/
│   ├── pipeline.py
│   ├── train.py
│   ├── predict.py
│   └── monitoring.py
├── tests/
├── scripts/
├── models/
├── logs/
├── data/
├── notebooks/
├── Dockerfile
├── Makefile
└── README.md