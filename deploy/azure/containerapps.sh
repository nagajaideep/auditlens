#!/usr/bin/env bash
# Deploy (or redeploy) the AuditLens API to Azure Container Apps. Safe to run repeatedly:
# existing resources are reused and the app is updated to the new image.
#
# Prereqs: `az login`, and a trained model in artifacts/models/bundle.joblib (`auditlens train`).
# Usage:   ACR=<globally-unique-lowercase-name> GEMINI_API_KEY=... ./deploy/azure/containerapps.sh
set -euo pipefail
: "${ACR:?set ACR to a globally unique, lowercase registry name, e.g. auditlens$RANDOM}"
RG=${RG:-auditlens-rg}
LOC=${LOC:-centralindia}
ENV_NAME=${ENV_NAME:-auditlens-env}
APP=${APP:-auditlens-api}
TAG=${TAG:-$(date +%Y%m%d%H%M%S)}
IMAGE="$ACR.azurecr.io/auditlens:$TAG"

test -f artifacts/models/bundle.joblib || { echo "No model bundle - run 'auditlens train' first"; exit 1; }

az extension add --name containerapp --upgrade -y --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az provider register --namespace Microsoft.ContainerRegistry --wait

az group create -n "$RG" -l "$LOC" -o none
az acr show -n "$ACR" -o none 2>/dev/null || az acr create -g "$RG" -n "$ACR" --sku Basic --admin-enabled true -o none
az acr build -r "$ACR" -t "auditlens:$TAG" -t auditlens:latest .   # builds in Azure, no local Docker needed

az containerapp env show -g "$RG" -n "$ENV_NAME" -o none 2>/dev/null || \
  az containerapp env create -g "$RG" -n "$ENV_NAME" -l "$LOC" -o none

ACR_USER=$(az acr credential show -n "$ACR" --query username -o tsv)
ACR_PASS=$(az acr credential show -n "$ACR" --query "passwords[0].value" -o tsv)

if az containerapp show -g "$RG" -n "$APP" -o none 2>/dev/null; then
  az containerapp secret set -g "$RG" -n "$APP" --secrets gemini-key="${GEMINI_API_KEY:-unset}" -o none
  az containerapp update -g "$RG" -n "$APP" --image "$IMAGE" \
    --set-env-vars GEMINI_API_KEY=secretref:gemini-key AUDITLENS_GENAI__LLM_PROVIDER=auto -o none
else
  az containerapp create -g "$RG" -n "$APP" --environment "$ENV_NAME" --image "$IMAGE" \
    --registry-server "$ACR.azurecr.io" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
    --target-port 8000 --ingress external --cpu 1.0 --memory 2.0Gi --min-replicas 0 --max-replicas 2 \
    --secrets gemini-key="${GEMINI_API_KEY:-unset}" \
    --env-vars GEMINI_API_KEY=secretref:gemini-key AUDITLENS_GENAI__LLM_PROVIDER=auto -o none
fi

URL=$(az containerapp show -g "$RG" -n "$APP" --query properties.configuration.ingress.fqdn -o tsv)
echo "Deployed $IMAGE"
echo "API docs:  https://$URL/docs"
echo "Health:    https://$URL/health"
echo "Tear down: az group delete -n $RG --yes --no-wait"
