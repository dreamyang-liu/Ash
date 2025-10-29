
purge:
	kubectl delete namespace apps

deploy:
	kubectl apply -f rbac.yaml
	kubectl apply -f infra.yaml
	kubectl apply -f stateless-mcps.yaml

.PHONY: deploy purge