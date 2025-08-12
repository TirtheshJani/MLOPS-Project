# syntax=docker/dockerfile:1

# ======================
# Stage 1: Builder
# ======================
FROM python:3.11 AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

# Copy dependency list and install them into an isolated prefix
COPY clinical-note-summarizer/requirements.txt /build/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefix=/install -r /build/requirements.txt


# ======================
# Stage 2: Final runtime
# ======================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create a non-root user and group for running the app
RUN groupadd --system app && \
    useradd --system --gid app --create-home --home-dir /home/app appuser

WORKDIR /app

# Copy only installed python packages from builder into standard location
COPY --from=builder /install /usr/local

# Copy application source code (models are optional and may be loaded remotely via MODEL_DIR)
COPY --chown=appuser:app clinical-note-summarizer/app/ /app/

# Switch to non-root user
USER appuser

# Expose default HTTP port
EXPOSE 8000

# Run the application with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


