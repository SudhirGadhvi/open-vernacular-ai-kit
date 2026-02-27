# Deploy Guide

This page covers minimal deployment options for the FastAPI wrapper.

## Docker image (local build)

```bash
docker build -t ovak-api:local .
docker run --rm -p 8000:8000 ovak-api:local
```

## Published image (GHCR)

A GitHub Actions workflow publishes images on release tags:

- Workflow: `.github/workflows/docker.yml`
- Image: `ghcr.io/<owner>/<repo>:<tag>`

## Render (minimal)

1. Create a new Web Service.
2. Use Docker deployment from this repository.
3. Expose port `8000`.
4. Health check path: `/healthz`.

## Fly.io (minimal)

1. Install `flyctl` and authenticate.
2. Initialize app:

```bash
fly launch --no-deploy
```

3. Deploy:

```bash
fly deploy
```

4. Set health checks to `/healthz`.

## Kubernetes (minimal)

Example deployment using a published image:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ovak-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ovak-api
  template:
    metadata:
      labels:
        app: ovak-api
    spec:
      containers:
        - name: ovak-api
          image: ghcr.io/<owner>/<repo>:<tag>
          ports:
            - containerPort: 8000
          livenessProbe:
            httpGet:
              path: /healthz
              port: 8000
          readinessProbe:
            httpGet:
              path: /healthz
              port: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: ovak-api
spec:
  selector:
    app: ovak-api
  ports:
    - port: 80
      targetPort: 8000
```
