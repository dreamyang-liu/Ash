
purge:
	kubectl delete namespace ash

deploy:
	kubectl apply -f rbac.yaml
	kubectl apply -f infra.yaml

.PHONY: deploy purge