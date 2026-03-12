FROM python:3.12-slim
WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

#Copy dependency files first for better caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the container
# --frozen ensures we get the exact versions from uv.lock
RUN uv sync --frozen --no-cache

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI will run on
EXPOSE 8000

# Use 'uv run' to start the FastAPI application
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
