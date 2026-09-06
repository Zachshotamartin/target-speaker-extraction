FROM python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.8.9 /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
ENV TSE_CHECKPOINT=/model/model.pt
CMD ["/app/.venv/bin/python", "-c", "import uvicorn; from tse.api import create_app; uvicorn.run(create_app(device='cpu'), host='0.0.0.0', port=8000)"]
