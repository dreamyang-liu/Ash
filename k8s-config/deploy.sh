#!/usr/bin/env bash
set -euo pipefail

# Deploy Ash infrastructure to Kubernetes.
# Usage:
#   ./deploy.sh              Deploy to current kubectl context (EKS)
#   ./deploy.sh --local      Deploy to minikube

NS="ash"
LOCAL=false

if [[ "${1:-}" == "--local" ]]; then
  LOCAL=true
  KUBECTL="minikube kubectl --"
  INFRA_FILE="infra-local.yaml"
else
  KUBECTL="kubectl"
  INFRA_FILE="infra.yaml"
fi

echo "==> Deploying Ash to namespace '$NS' (local=$LOCAL)"

# Create namespace and RBAC
$KUBECTL apply -f rbac.yaml

# Deploy infrastructure (Redis, Control Plane, Gateway)
$KUBECTL apply -f "$INFRA_FILE"

# Wait for rollouts
echo "==> Waiting for deployments..."
$KUBECTL -n $NS rollout status deploy/redis --timeout=60s
$KUBECTL -n $NS rollout status deploy/control-plane --timeout=90s
$KUBECTL -n $NS rollout status deploy/gateway --timeout=90s

echo "==> All deployments ready."

# Print service URLs
if $LOCAL; then
  echo ""
  echo "Service URLs (minikube):"
  minikube service control-plane -n $NS --url
  minikube service gateway -n $NS --url
else
  echo ""
  echo "Services:"
  $KUBECTL -n $NS get svc -o wide
fi
