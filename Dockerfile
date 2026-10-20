
# ============================================================
# MILESTONE 10 - FRAUD DETECTION API CONTAINER
# Step 10.09 - CPU-Only Dockerfile
# ============================================================

# ------------------------------------------------------------
# 1. Use Python 3.11 on a lightweight Debian image.
# ------------------------------------------------------------

FROM python:3.11-slim-bookworm

# ------------------------------------------------------------
# 2. Configure Python and pip.
# ------------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt/fraud-api

# ------------------------------------------------------------
# 3. Install required system runtime libraries.
# ------------------------------------------------------------

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# 4. Install pinned Python dependencies.
# ------------------------------------------------------------

COPY requirements.txt ./requirements.txt

RUN python -m pip install \
    --no-cache-dir \
    -r requirements.txt

# ------------------------------------------------------------
# 5. Create a non-root application user.
# ------------------------------------------------------------

RUN groupadd --system --gid 10001 fraudapi \
    && useradd --system \
       --uid 10001 \
       --gid fraudapi \
       --home-dir /opt/fraud-api \
       --no-create-home \
       fraudapi

# ------------------------------------------------------------
# 6. Copy application source and configuration.
# ------------------------------------------------------------

COPY --chown=fraudapi:fraudapi app/ ./app/
COPY --chown=fraudapi:fraudapi src/ ./src/
COPY --chown=fraudapi:fraudapi configs/ ./configs/

# ------------------------------------------------------------
# 7. Prepare an external artifact mount location.
# ------------------------------------------------------------

RUN mkdir -p /opt/fraud-api/artifacts/milestone_07 \
    && chown -R fraudapi:fraudapi /opt/fraud-api

# ------------------------------------------------------------
# 8. Drop root privileges.
# ------------------------------------------------------------

USER fraudapi

# ------------------------------------------------------------
# 9. Document the container's listening port.
# ------------------------------------------------------------

EXPOSE 8000

# ------------------------------------------------------------
# 10. Verify model readiness with Docker health checks.
# ------------------------------------------------------------

HEALTHCHECK --interval=30s \
    --timeout=5s \
    --start-period=90s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3)" || exit 1

# ------------------------------------------------------------
# 11. Start the FastAPI inference service.
# ------------------------------------------------------------

CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]
