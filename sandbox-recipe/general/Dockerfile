FROM python:3.13-slim AS builder

# Create virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastmcp terminal-controller ddgs mcp[cli] mcp-server-fetch

FROM python:3.13-slim

# Set locale
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Copy virtual environment from builder
ENV VIRTUAL_ENV=/opt/venv
COPY --from=builder $VIRTUAL_ENV $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/
COPY ddgs_mcp /ddgs_mcp
# Install only essential runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        netcat-traditional && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY main.py /app/main.py

EXPOSE 3000

ENTRYPOINT ["python", "/app/main.py"]
