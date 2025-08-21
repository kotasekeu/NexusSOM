# Base image s Pythonem
FROM python:3.12

WORKDIR /app

COPY ./requirements.txt ./
RUN python3 -m venv /venv && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/venv/bin:$PATH"

COPY . .

ENV PYTHONPATH=/app

CMD ["python3", "main.py"]