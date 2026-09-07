FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt pytest xgboost

COPY . .

# Sanity-check the install by default (no network / PhysioNet access needed).
CMD ["python", "preprocess_apnea_ecg.py", "--self-test", "--out-dir", "/app/out"]
