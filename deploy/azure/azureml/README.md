# Optional: Azure ML managed online endpoint

Container Apps (`../containerapps.sh`) is the simpler and cheaper route. Use Azure ML if you want the
model registry and managed endpoints:

```bash
az extension add -n ml
az ml workspace create -g auditlens-rg -n auditlens-ws
az ml environment create -f environment.yml -g auditlens-rg -w auditlens-ws
az ml online-endpoint create -f endpoint.yml -g auditlens-rg -w auditlens-ws
az ml online-deployment create -f deployment.yml -g auditlens-rg -w auditlens-ws --all-traffic
```

The deployment runs the same Docker image (pushed to ACR) and serves FastAPI on port 8000.
