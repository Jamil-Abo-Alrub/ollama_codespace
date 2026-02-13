FROM python:3.13-slim

WORKDIR /app

COPY  mlruns .

COPY mlflow.db .

COPY StudentsPerformance.csv .

RUN pip install --no-cache-dir mlflow pandas scikit-learn

EXPOSE 8080

CMD mlflow server --port 8080 --host 0.0.0.0 --allowed-hosts "*"