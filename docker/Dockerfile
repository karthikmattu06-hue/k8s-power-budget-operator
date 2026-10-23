# One image, three programs: pbo-mockserve, pbo-emulator, pbo-loadgen.
# The Kubernetes manifests pick which one runs via `command:`.
#
# Build:  docker build -f docker/Dockerfile -t pbo:dev .
# kind:   kind load docker-image pbo:dev --name pbo   (copies it into every kind node)
FROM python:3.12-slim

# No .pyc files, unbuffered logs (so `kubectl logs` shows lines immediately).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PBO_CURVES=/app/curves/lever-curves-v0.3.json

WORKDIR /app
# Copy only what the install needs first, so Docker caches the dependency layer.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[serve]"
COPY curves ./curves

# Run as an unprivileged user; nothing here needs root.
RUN useradd --uid 10001 --no-create-home pbo
USER 10001

CMD ["pbo-mockserve"]
