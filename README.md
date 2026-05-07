# Housing Price Prediction MLOps MVP

This project trains and serves a machine learning model for the Boston Housing dataset.

It includes:

- EDA notebooks
- sklearn preprocessing pipeline
- RandomForest regression model
- FastAPI backend
- Streamlit frontend
- pytest tests
- Docker support

## Project Structure

```text
housing-mlops/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── data/
│   └── raw/
│       └── HousingData.csv
├── models/
├── src/
│   ├── pipeline.py
│   ├── train.py
│   └── predict.py
├── tests/
│   └── test_pipeline.py
├── Dockerfile
├── Makefile
├── pyproject.toml
└── README.md