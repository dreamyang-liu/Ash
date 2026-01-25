all:
	cd k8s-scaffold && $(MAKE)
	cd sandbox-recipe && $(MAKE)

clean:
	cd scaffold && $(MAKE) clean


build-image-local: start-minikube
	cd k8s-scaffold/control-plane && minikube image build -f Dockerfile -t rl-sandbox-cp:0.1 .
	cd k8s-scaffold/gateway && minikube image build -f Dockerfile -t rl-sandbox-gateway:0.1 .
	cd sandbox-recipe/general && minikube image build -f Dockerfile -t sandbox:general-0.1 .

apply-config-local: build-image-local start-minikube
	cd k8s-config && \
		minikube kubectl -- apply -f rbac.yaml && \
		minikube kubectl -- apply -f infra-local.yaml && \
		minikube kubectl -- apply -f stateless-mcps.yaml && \
		minikube kubectl -- -n ash rollout status deploy/gateway && \
		minikube kubectl -- -n ash rollout status deploy/control-plane

start-minikube:
	minikube start
	minikube kubectl -- apply -f local-lb.yaml

all-local: build-local apply-config-local 

.PHONY: all clean build-local apply-config-local start-minikube