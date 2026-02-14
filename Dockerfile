FROM python:3.10-slim

WORKDIR /app

COPY src/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/

ENV DATASET=breast_cancer \
    MODEL_PATH=/app/artifacts/model.pkl \
    REPORT_PATH=/app/artifacts/report.json

RUN mkdir -p /app/artifacts

CMD ["python", "main.py"]
