# Kubernetes (minikube)

```bash
minikube start --cpus 4 --memory 6g
docker build -t auditlens:latest .        # after `auditlens train` so the bundle is baked in
minikube image load auditlens:latest
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/
kubectl -n auditlens rollout status deploy/auditlens-api
curl "$(minikube service auditlens-api -n auditlens --url)/health"
```
