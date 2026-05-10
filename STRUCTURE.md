# Project Structure

automl-platform/
├── app/                        # Core application logic
│   ├── main.py                 # Streamlit UI (entry point)
│   ├── ingest.py               # Data loading (CSV, Excel, SQL)
│   ├── preprocess.py           # Auto sklearn preprocessing pipeline
│   ├── automl.py               # Model benchmarking engine
│   ├── explain.py              # Feature importance & confusion matrix charts
│   ├── insights.py             # Claude API — AI narrative summaries
│   ├── visualize.py            # Reusable matplotlib/seaborn chart helpers
│   └── utils.py                # Shared utilities (feature names, task detection)
│
├── api/                        # FastAPI model serving
│   ├── serve.py                # /predict and /health endpoints
│   └── schemas.py              # Pydantic request/response models
│
├── data/
│   ├── raw/                    # Original uploaded datasets (gitignored)
│   ├── processed/              # Cleaned/transformed data (gitignored)
│   └── sample/                 # Sample datasets for testing
│
├── models/
│   ├── saved/                  # Serialised .pkl model files (gitignored)
│   └── exports/                # ONNX / joblib exports for deployment
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb # Full training walkthrough
│
├── tests/
│   ├── test_ingest.py          # Unit tests for data loading
│   ├── test_preprocess.py      # Unit tests for pipeline
│   └── test_automl.py          # Integration test for model training
│
├── reports/                    # Auto-generated HTML/PDF reports
├── assets/                     # Logos, icons, static files
├── mlflow_tracking/            # MLflow experiment logs (gitignored)
│
├── Dockerfile
├── docker-compose.yml          # Streamlit + FastAPI + MLflow services
├── requirements.txt
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
└── STRUCTURE.md                # This file
