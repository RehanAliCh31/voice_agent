# The server, with the real engines available.
#
# Two stages so the toolchain that builds the wheels does not ship with
# them. CPU wheels for torch throughout: nothing here uses a GPU, and the
# CUDA build is an order of magnitude larger.
#
# The model endpoint and the PostgreSQL bank are both reached over the
# network at runtime, so a container needs to be able to see them:
#
#     docker build -t voicebot .
#     docker run --rm -p 8000:8000 \
#       -e VOICEBOT_CONFIG=configs/serve.yaml \
#       --env-file .env -e VOICEBOT_DB_HOST=host.docker.internal \
#       -v voicebot-data:/app/data \
#       voicebot
#
# `--env-file .env` passes the database settings in at run time; .env is in
# .dockerignore, so the password is never baked into the image.
# `host.docker.internal` is the machine running Docker, which is where a
# local PostgreSQL is -- inside the container, localhost is the container.
# PostgreSQL must then accept that connection: `listen_addresses` in
# postgresql.conf and a host line in pg_hba.conf.
#
# The volume holds the downloaded models, and the SQLite file if the config
# is switched back to `engine: sqlite`.

FROM python:3.12-slim AS build

WORKDIR /app
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml constraints.txt README.md ./
COPY src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ".[serve]" -c constraints.txt

FROM python:3.12-slim

# Not root. The server writes only to data/, which is a volume.
RUN useradd --create-home --uid 10001 voicebot
WORKDIR /app

COPY --from=build /opt/venv /opt/venv
COPY --chown=voicebot:voicebot configs ./configs
COPY --chown=voicebot:voicebot src ./src

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/data/models

RUN mkdir -p /app/data && chown voicebot:voicebot /app/data
USER voicebot

EXPOSE 8000

# Readiness, not liveness: /ready answers only once the engines have loaded,
# and Whisper takes a while on first run while it downloads.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=4)"

CMD ["uvicorn", "voicebot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
