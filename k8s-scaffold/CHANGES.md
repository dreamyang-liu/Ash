# K8s-Scaffold Improvements

This document summarizes the improvements made to the k8s-scaffold repository.

## Changes Implemented

### 1. Fixed Dockerfile Healthcheck Ports ✓
- **Files**: `control-plane/Dockerfile`, `gateway/Dockerfile`
- **Change**: Updated healthcheck from `localhost:80` to `localhost:8080` to match actual listening port
- Both services listen on `:8080`, not `:80`

### 2. Redis Record Enhancement ✓
- **Added Fields**: `name` and `namespace` to Redis records
- **Location**: `store/redis.go`, `handler/spawn.go`
- **Backwards Compatibility**: The destroy handler falls back to hostname parsing if `name`/`namespace` fields are missing
- Records now store:
  - `uuid` - Unique sandbox identifier
  - `host` - Full DNS hostname (e.g., `service.namespace.svc.cluster.local`)
  - `port` - Service port
  - `status` - Sandbox status (ready/starting/exists)
  - `name` - Service/deployment name (NEW)
  - `namespace` - Kubernetes namespace (NEW)

### 3. Control-Plane Refactoring ✓
Split `control-plane/main.go` (618 lines) into focused packages:

```
control-plane/
├── main.go              # Startup, router, graceful shutdown (108 lines)
├── config/
│   └── config.go        # Configuration loading
├── k8s/
│   ├── client.go        # Kubernetes client creation
│   └── sandbox.go       # Sandbox CRUD operations
├── store/
│   └── redis.go         # Redis operations with metrics
├── handler/
│   ├── spawn.go         # POST /create handler
│   └── destroy.go       # DELETE /destroy handler
├── metrics/
│   └── metrics.go       # Prometheus metrics definitions
└── reconciler/
    └── reconciler.go    # Background reconciliation loop
```

**Benefits**:
- Clear separation of concerns
- Easier testing (dependencies injected via `handler.Dependencies`)
- No global state (except for metrics)
- Better code organization by responsibility

### 4. Removed (Skipped #4)
Change #4 was skipped in the original numbering.

### 5. Idempotent K8s Operations ✓
- **File**: `k8s/sandbox.go`
- **Change**: Handle `AlreadyExists` errors gracefully
- When creating a Deployment or Service that already exists:
  - Catch the error using `errors.IsAlreadyExists(err)`
  - Fetch the existing resource instead of failing
  - Return status `"exists"` to indicate idempotent behavior
  - Continue normally (no 500 error)

### 6. Reconciliation Loop ✓
- **File**: `reconciler/reconciler.go`
- **Configuration**: `RECONCILE_INTERVAL` environment variable (default: 30 seconds)
- **Started**: In `main.go` as a background goroutine

**Reconciliation Tasks**:

1. **K8s → Redis Sync (Self-Healing)**:
   - Lists all Deployments with labels `from=control-plane,type=sandbox`
   - For each deployment, checks if a corresponding Redis record exists
   - If missing, creates a Redis record with a generated UUID
   - Recovers from Redis data loss

2. **Redis → K8s Sync (Cleanup)**:
   - Lists all Redis records matching `sandbox:*`
   - For each record, checks if the K8s deployment still exists
   - If not found, deletes the stale Redis record
   - Prevents Redis key accumulation

3. **Metrics Update**:
   - Updates `sandbox_active_count` gauge with current sandbox count

### 7. Prometheus Metrics ✓
- **File**: `metrics/metrics.go`
- **Endpoint**: `GET /metrics`
- **Dependency**: `github.com/prometheus/client_golang v1.19.0`

**Metrics Tracked**:

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `sandbox_create_total` | Counter | `status` (success/error) | Total sandbox creation attempts |
| `sandbox_destroy_total` | Counter | `status` (success/error) | Total sandbox destruction attempts |
| `sandbox_create_duration_seconds` | Histogram | - | Duration of sandbox creation |
| `sandbox_active_count` | Gauge | - | Current number of active sandboxes |
| `redis_operation_duration_seconds` | Histogram | `operation` (hset/hgetall/del/scan) | Redis operation duration |

**Integration**:
- Metrics are recorded in handlers (`handler/spawn.go`, `handler/destroy.go`)
- Redis metrics are recorded in store package (`store/redis.go`)
- Active count updated by reconciler (`reconciler/reconciler.go`)

## Module Information
- **Module Name**: `github.com/rl-sandbox/k8s-cp` (unchanged)
- **Go Version**: 1.24.3

## Testing
To verify the changes:

1. Build the control-plane service:
   ```bash
   cd control-plane
   go mod tidy
   go build -o k8s-cp .
   ```

2. Build with Docker:
   ```bash
   docker build -t control-plane:latest ./control-plane
   docker build -t gateway:latest ./gateway
   ```

3. Check metrics endpoint:
   ```bash
   curl http://localhost:8080/metrics
   ```

4. Test idempotent create:
   ```bash
   # Create once
   curl -X POST http://localhost:8080/create -d '{"image": "nginx:latest"}'
   # Create again with same name - should return "exists" status
   curl -X POST http://localhost:8080/create -d '{"image": "nginx:latest", "name": "test-sandbox"}'
   ```

## Gateway Changes
- **Only Change**: Fixed Dockerfile healthcheck port from 80 to 8080
- No code refactoring needed for gateway (as requested)

## Backwards Compatibility
- Old Redis records (without `name`/`namespace` fields) still work
- Destroy handler falls back to hostname parsing for old records
- All existing API contracts maintained
- No breaking changes to request/response formats
