# ВкладНавигатор — образ веб-формы ассистента (развёртывание как MVP).
FROM python:3.11-slim

WORKDIR /app

# Сначала зависимости — лучше кэшируется при изменениях кода.
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY data ./data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/healthz')"

CMD ["streamlit", "run", "app/streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
