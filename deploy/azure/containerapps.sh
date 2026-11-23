#!/usr/bin/env bash
# Deploy the AuditLens API to Azure Container Apps (fits Azure for Students credits).
# Prereqs: `az login`, Docker, and a trained bundle in artifacts/models/ (run `auditlens train`).
set -euo pipefail
RG=${RG:-auditlens-rg}
LOC=${LOC:-centralindia}
ACR=${ACR:-auditlens$RANDOM}          # must be globally unique, lowercase
ENV_NAME=${ENV_NAME:-auditlens-env}
APP=${APP:-auditlens-api}

az extension add --name containerapp --upgrade -y
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az group create -n "$RG" -l "$LOC"
az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true
az acr build -r "$ACR" -t auditlens:latest .           # builds in Azure, no local Docker push needed

az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOC"
az containerapp create -g "$RG" -n "$APP" --environment "$ENV_NAME" \
  --image "$ACR.azurecr.io/auditlens:latest" --registry-server "$ACR.azurecr.io" \
  --target-port 8000 --ingress external --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 2 \
  --secrets gemini-key="${GEMINI_API_KEY:-unset}" \
  --env-vars GEMINI_API_KEY=secretref:gemini-key AUDITLENS_GENAI__LLM_PROVIDER=auto

URL=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
echo "Deployed: https://$URL/docs"
echo "Tear down when done: az group delete -n $RG --yes --no-wait"
