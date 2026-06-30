#!/bin/bash
set -e

echo "=== Verifying k8s-scaffold improvements ==="
echo ""

echo "1. Checking control-plane structure..."
cd control-plane

required_files=(
    "main.go"
    "config/config.go"
    "k8s/client.go"
    "k8s/sandbox.go"
    "store/redis.go"
    "handler/spawn.go"
    "handler/destroy.go"
    "metrics/metrics.go"
    "reconciler/reconciler.go"
    "Dockerfile"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

echo ""
echo "2. Checking Dockerfile healthcheck..."
if grep -q "localhost:8080/healthz" Dockerfile; then
    echo "✓ Control-plane Dockerfile uses correct port 8080"
else
    echo "✗ Control-plane Dockerfile healthcheck port incorrect"
    exit 1
fi

cd ../gateway
if grep -q "localhost:8080/healthz" Dockerfile; then
    echo "✓ Gateway Dockerfile uses correct port 8080"
else
    echo "✗ Gateway Dockerfile healthcheck port incorrect"
    exit 1
fi

echo ""
echo "3. Checking for Redis name/namespace fields..."
cd ../control-plane
if grep -q '"name":' store/redis.go && grep -q '"namespace":' store/redis.go; then
    echo "✓ Redis record includes name and namespace fields"
else
    echo "✗ Redis record missing name/namespace fields"
    exit 1
fi

echo ""
echo "4. Checking for idempotent create (AlreadyExists handling)..."
if grep -q "IsAlreadyExists" k8s/sandbox.go; then
    echo "✓ Idempotent create implemented"
else
    echo "✗ Idempotent create not implemented"
    exit 1
fi

echo ""
echo "5. Checking for reconciler..."
if [ -f "reconciler/reconciler.go" ]; then
    echo "✓ Reconciler package exists"
else
    echo "✗ Reconciler missing"
    exit 1
fi

echo ""
echo "6. Checking for Prometheus metrics..."
if grep -q "prometheus" go.mod && [ -f "metrics/metrics.go" ]; then
    echo "✓ Prometheus metrics implemented"
else
    echo "✗ Prometheus metrics missing"
    exit 1
fi

echo ""
echo "7. Checking for metrics endpoint in main..."
if grep -q "/metrics" main.go; then
    echo "✓ Metrics endpoint registered"
else
    echo "✗ Metrics endpoint not registered"
    exit 1
fi

echo ""
echo "=== All verifications passed! ==="
echo ""
echo "To build and test:"
echo "  cd control-plane && go mod tidy && go build -o k8s-cp ."
echo "  cd ../gateway && go build -o k8s-gateway ."
echo ""
echo "To build Docker images:"
echo "  docker build -t control-plane:latest ./control-plane"
echo "  docker build -t gateway:latest ./gateway"
