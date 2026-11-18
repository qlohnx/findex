FROM python:3.14-slim-trixie AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/
ENV UV_NO_DEV=1 UV_PYTHON_DOWNLOADS=0 UV_NO_CACHE=1
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/
COPY demo/corpus/ ./demo/corpus/
RUN uv sync --locked --no-editable
RUN /app/.venv/bin/findex index /app/demo/corpus --out /app/demo/index.json --positions

FROM python:3.14-slim-trixie
WORKDIR /app
RUN useradd --system --uid 10001 --create-home appuser
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/demo/index.json /app/demo/index.json
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 INDEX_PATH=/app/demo/index.json
USER appuser
EXPOSE 8000
CMD ["findex", "serve", "--host", "0.0.0.0"]
