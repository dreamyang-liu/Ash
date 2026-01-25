build: build-control-plane build-gateway

build-control-plane:
	cd control-plane && docker build -t timemagic/ash:control-plane-0.1 .
build-gateway:
	cd gateway && docker build -t timemagic/ash:gateway-0.1 .

clean:
	docker rmi timemagic/ash:gateway-0.1
	docker rmi timemagic/ash:control-plane-0.1

.PHONY: build build-control-plane build-gateway clean
