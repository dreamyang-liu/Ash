# Quick Start Guide

## Prerequisites
- Go 1.24.3 or later
- Docker (for containerized deployment)
- Kubernetes cluster access
- Redis instance

## Build from Source

```bash
# Control plane
cd control-plane
go mod tidy
go build -o k8s-cp .

# Gateway
cd ../gateway
go build -o k8s-gateway .
```

## Build Docker Images

```bash
# From repository root
docker build -t control-plane:latest ./control-plane
docker build -t gateway:latest ./gateway
```

## Configuration

### Control Plane Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TARGET_NAMESPACE` | `ash` | K8s namespace for sandboxes |
| `REDIS_HOST` | `localhost` | Redis hostname |
| `REDIS_PORT` | `6379` | Redis port |
| `REDIS_DB` | `0` | Redis database number |
| `WAIT_DEPLOY_READY_SEC` | `120` | Timeout for deployment readiness |
| `WAIT_SVC_IP_SEC` | `120` | Timeout for service IP assignment |
| `SERVICE_ACCOUNT_NAME` | `default` | K8s service account for pods |
| `RECONCILE_INTERVAL` | `30` | Reconciler interval (seconds) |

### Gateway Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LISTEN_ADDR` | `:8080` | Gateway listen address |
| `SANDBOX_HEADER` | `X-Sandbox-ID` | Header for sandbox UUID |
| `REDIS_ADDR` | `127.0.0.1:6379` | Redis address |
| `REDIS_DB` | `0` | Redis database number |
| `DEFAULT_SCHEME` | `http` | Upstream protocol |

## Running Locally

### Start Control Plane
```bash
export TARGET_NAMESPACE=ash
export REDIS_HOST=localhost
export REDIS_PORT=6379

./k8s-cp
# Listening on :8080
```

### Start Gateway
```bash
export REDIS_ADDR=localhost:6379
export SANDBOX_HEADER=X-Sandbox-ID

./k8s-gateway
# Listening on :8080
```

## API Examples

### Create a Sandbox
```bash
curl -X POST http://localhost:8080/create \
  -H "Content-Type: application/json" \
  -d '{
    "image": "nginx:latest",
    "name": "my-sandbox",
    "ports": [{"container_port": 80}],
    "env": {"DEBUG": "true"},
    "resources": {
      "requests": {"cpu": "100m", "memory": "128Mi"},
      "limits": {"cpu": "500m", "memory": "512Mi"}
    }
  }'
```

Response:
```json
{
  "name": "my-sandbox",
  "uuid": "my-sandbox-abc123-uuid",
  "namespace": "ash",
  "status": "Ready",
  "service_type": "ClusterIP",
  "cluster_ip": "10.96.1.23",
  "host": "my-sandbox.ash.svc.cluster.local",
  "ports": [80]
}
```

### Destroy Specific Sandboxes
```bash
curl -X DELETE http://localhost:8080/destroy \
  -H "Content-Type: application/json" \
  -d '{"ids": ["my-sandbox-abc123-uuid"]}'
```

### Destroy All Sandboxes
```bash
curl -X DELETE http://localhost:8080/destroy \
  -H "Content-Type: application/json" \
  -d '{"all": true}'
```

### Health Check
```bash
curl http://localhost:8080/healthz
# ok

curl http://localhost:8080/readyz
# ready
```

### Metrics
```bash
curl http://localhost:8080/metrics
```

## Kubernetes Deployment

### Deploy Control Plane
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: control-plane
  namespace: ash
spec:
  replicas: 1
  selector:
    matchLabels:
      app: control-plane
  template:
    metadata:
      labels:
        app: control-plane
    spec:
      serviceAccountName: control-plane
      containers:
      - name: control-plane
        image: control-plane:latest
        ports:
        - containerPort: 8080
        env:
        - name: TARGET_NAMESPACE
          value: "ash"
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        - name: RECONCILE_INTERVAL
          value: "30"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: control-plane
  namespace: ash
spec:
  selector:
    app: control-plane
  ports:
  - port: 8080
    targetPort: 8080
```

### Deploy Gateway
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gateway
  namespace: ash
spec:
  replicas: 2
  selector:
    matchLabels:
      app: gateway
  template:
    metadata:
      labels:
        app: gateway
    spec:
      containers:
      - name: gateway
        image: gateway:latest
        ports:
        - containerPort: 8080
        env:
        - name: REDIS_ADDR
          value: "redis-service:6379"
        - name: SANDBOX_HEADER
          value: "X-Sandbox-ID"
        livenessProbe:
          httpGet:
            path: /healthz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /readyz
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: gateway
  namespace: ash
spec:
  selector:
    app: gateway
  ports:
  - port: 80
    targetPort: 8080
```

## Monitoring Setup

### Prometheus Scrape Config
```yaml
scrape_configs:
  - job_name: 'control-plane'
    static_configs:
      - targets: ['control-plane:8080']
    metrics_path: '/metrics'
```

### Grafana Dashboard Query Examples
```promql
# Total sandbox creations
rate(sandbox_create_total[5m])

# Success rate
rate(sandbox_create_total{status="success"}[5m]) / rate(sandbox_create_total[5m])

# Average creation time
rate(sandbox_create_duration_seconds_sum[5m]) / rate(sandbox_create_duration_seconds_count[5m])

# Active sandboxes
sandbox_active_count

# Redis operation latency (95th percentile)
histogram_quantile(0.95, rate(redis_operation_duration_seconds_bucket[5m]))
```

## Troubleshooting

### Control Plane Won't Start
1. Check Redis connection: `redis-cli -h <redis-host> ping`
2. Check K8s permissions: `kubectl auth can-i create deployments --namespace=ash`
3. Check logs: `kubectl logs -n ash deployment/control-plane`

### Sandbox Not Ready
1. Check deployment status: `kubectl get deployment -n ash <sandbox-name>`
2. Check pod logs: `kubectl logs -n ash <pod-name>`
3. Check readiness probe: Ensure container listens on the specified port

### Reconciler Not Working
1. Check reconciler logs in control-plane logs
2. Verify `RECONCILE_INTERVAL` is set correctly
3. Check K8s API access: `kubectl get deployments -n ash -l from=control-plane`

### Gateway 404 Errors
1. Verify Redis record exists: `redis-cli hgetall sandbox:<uuid>`
2. Check header is correct: Use `X-Sandbox-ID` by default
3. Verify service DNS: `nslookup <service-name>.<namespace>.svc.cluster.local`

## Verification

Run the automated verification script:
```bash
./verify-build.sh
```

Expected output: All checks should pass with ✓ marks.

## Next Steps
- Set up Prometheus monitoring
- Create Grafana dashboards
- Configure alerting rules
- Add authentication to API endpoints
- Implement rate limiting
