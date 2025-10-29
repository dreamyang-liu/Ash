build: build-control-plane build-gateway

build-control-plane:
	cd control-plane && docker build -t k8s-control-plane .
build-gateway:
	cd gateway && docker build -t k8s-gateway .

clean:
	docker rmi k8s-gateway
	docker rmi k8s-control-plane

.PHONY: build build-control-plane build-gateway clean
