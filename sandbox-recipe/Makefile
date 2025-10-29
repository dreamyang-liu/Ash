build: build-general

build-general:
	cd general && docker build -t sandbox-general .

clean:
	docker rmi sandbox-general
	docker system prune

.PHONY: build build-general