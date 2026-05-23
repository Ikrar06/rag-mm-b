FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY automation_qa/requirements.txt ./automation_qa/requirements.txt
RUN python -m pip install --no-cache-dir -r automation_qa/requirements.txt

COPY automation_qa/ ./automation_qa/

CMD ["python", "-m", "automation_qa.sync_to_sheets", "--watch"]
