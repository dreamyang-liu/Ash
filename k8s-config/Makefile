NS := ash

deploy:
	./deploy.sh

deploy-local:
	./deploy.sh --local

status:
	kubectl -n $(NS) get pods -o wide
	kubectl -n $(NS) get svc

logs-cp:
	kubectl -n $(NS) logs -f -l app=control-plane --all-containers

logs-gw:
	kubectl -n $(NS) logs -f -l app=gateway --all-containers

purge:
	kubectl delete namespace $(NS) --ignore-not-found

.PHONY: deploy deploy-local status logs-cp logs-gw purge
