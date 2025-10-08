
minikube kubectl -- delete namespace apps



docker build -f Dockerfile.cp -t timemagic/rl-mcp:rl-sandbox-cp-0.1 .
docker build -f Dockerfile.gateway -t timemagic/rl-mcp:rl-sandbox-gateway-0.1 .

# minikube image build -f Dockerfile.cp -t rl-sandbox-cp:0.1 .
# minikube image build -f Dockerfile.gateway -t rl-sandbox-gateway:0.1 .

minikube kubectl -- apply -f rbac.yaml
minikube kubectl -- apply -f infra.yaml
minikube kubectl -- apply -f stateless-mcps.yaml


# minikube kubectl -- -n apps rollout restart deploy/spawner
minikube kubectl -- -n apps rollout status deploy/gateway
minikube kubectl -- -n apps rollout status deploy/control-plane

minikube service control-plane -n apps --url
minikube service gateway -n apps --url
minikube service proxy-mcp -n apps --url

# CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build
# minikube image build -f Dockerfile.gateway -t rl-sandbox-gateway:0.1 .
# minikube kubectl -- -n apps port-forward svc/spawner 8080:80


curl -X POST http://localhost:8080/spawn \
  -H "Content-Type: application/json" \
  -d '{
    "image": "nginx:1.27",
    "ports": [{"container_port": 80}],
    "expose": "LoadBalancer",
    "env": {"HELLO": "world"},
    "replicas": 1
  }'
# minikube kubectl -- -n apps logs -f -l app=spawner --all-containers --prefix --max-log-requests=20



# # 滚动状态（快速看到是否 ProgressDeadlineExceeded）
# minikube kubectl -- -n $NS rollout status deploy/$DEP --timeout=30s

# # Deployment 条件 & 事件
# minikube kubectl -- -n $NS describe deploy/$DEP | sed -n '/Conditions:/,/Events:/p'
# minikube kubectl -- -n $NS describe deploy/$DEP | sed -n '/Events:/,$p'

# # 关联 ReplicaSet（新旧版本各多少副本）
# minikube kubectl -- -n $NS get rs -l app=$APP -o wide --sort-by=.metadata.creationTimestamp

# # Pod 概览（状态/READY/REASON）
# minikube kubectl -- -n $NS get pod -l app=$APP -o wide

# # cd gateway
# # CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build
# # cd ..
# # minikube image build -f Dockerfile.gateway -t rl-sandbox-gateway:0.1 .
# # minikube kubectl -- -n apps rollout restart deploy/gateway
# # minikube kubectl -- -n apps rollout status deploy/gateway