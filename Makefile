all:
	cd k8s-scaffold && $(MAKE)

clean:
	cd k8s-scaffold && $(MAKE) clean

build-image-local: start-minikube
	cd k8s-scaffold/control-plane && minikube image build -f Dockerfile -t dreamyangliu/ash:control-plane-0.1 .
	cd k8s-scaffold/gateway && minikube image build -f Dockerfile -t dreamyangliu/ash:gateway-0.1 .

apply-config-local: build-image-local
	cd k8s-config && \
		minikube kubectl -- apply -f rbac.yaml && \
		minikube kubectl -- apply -f infra-local.yaml && \
		minikube kubectl -- -n ash rollout status deploy/gateway && \
		minikube kubectl -- -n ash rollout status deploy/control-plane

start-minikube:
	minikube start

all-local: start-minikube build-image-local apply-config-local

.PHONY: all clean build-image-local apply-config-local start-minikube all-local
