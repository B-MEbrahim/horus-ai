FROM python:3.12-slim

# bring in uv (the same tool you use locally)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 1) install libraries first; this layer is reused unless dependencies change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

# 2) then add your code
COPY src ./src
RUN uv sync --frozen --no-default-groups

ENV PATH="/app/.venv/bin:$PATH" \
    HORUS_WINDOW_SIZE=30

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "horus_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]