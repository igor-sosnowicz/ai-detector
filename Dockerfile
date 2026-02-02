FROM astral/uv:python3.13-trixie

ENV UV_COMPILE_BYTECODE=1
ENV UV_HTTP_TIMEOUT=300

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --frozen

COPY . .

EXPOSE ${API_PORT}

HEALTHCHECK CMD uv run --no-dev ai-detector --health || exit 1

CMD ["uv", "run", "--frozen", "--no-dev", "ai-detector"]
