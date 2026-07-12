# OpenJustice.ai — Azure Infrastructure

## Overview

This directory contains Bicep infrastructure-as-code templates for deploying OpenJustice.ai on Azure. The templates provision a production-grade infrastructure including AKS, ACR, PostgreSQL, Blob Storage, Key Vault, and monitoring.

## Architecture

```
Azure Front Door (CDN + WAF)
       │
┌──────┴──────┐
│ Azure Static│ ← Frontend (React + Vite)
│  Web Apps   │
└──────┬──────┘
       │ /api/*
┌──────┴──────┐
│    AKS      │ ← Backend API (FastAPI)
│ K8s Cluster │
└──────┬──────┘
       │
┌──────┼──────────┐
│      │          │
│  PostgreSQL   Redis   │
│  (Auth/Users) (Cache) │
│      │          │
│  Elasticsearch Milvus  │
│  (BM25)      (Vector) │
│      │          │
│  Azure Blob Storage    │
└────────────────────┘
```

## Prerequisites

- [Azure CLI](https://docs.microsoft.com/cli/azure/install-azure-cli) >= 2.60
- [Bicep CLI](https://docs.microsoft.com/azure/azure-resource-manager/bicep/install)
- Contributor or Owner access to an Azure subscription

## Deployment Instructions

### 1. Login and Set Subscription

```bash
az login
az account set --subscription "<subscription-id>"
```

### 2. Deploy Core Infrastructure

```bash
# Preview the deployment
az deployment sub what-if \
  --name openjustice-preview \
  --location eastus \
  --template-file infrastructure/main.bicep \
  --parameters environment=dev

# Deploy to dev
az deployment sub create \
  --name openjustice-dev-deploy \
  --location eastus \
  --template-file infrastructure/main.bicep \
  --parameters environment=dev

# Deploy to production
az deployment sub create \
  --name openjustice-prod-deploy \
  --location eastus \
  --template-file infrastructure/main.bicep \
  --parameters environment=prod
```

### 3. Deploy Monitoring

```bash
# Get outputs from main deployment
az deployment sub show \
  --name openjustice-dev-deploy \
  --query properties.outputs

# Deploy monitoring alerts
az deployment group create \
  --resource-group rg-openjustice-dev \
  --template-file infrastructure/monitoring.bicep \
  --parameters \
    environment=dev \
    appInsightsConnectionString="<from-outputs>" \
    logAnalyticsWorkspaceId="<from-outputs>"
```

### 4. Configure Key Vault Secrets

After deployment, populate Key Vault with required secrets:

```bash
# Set secrets via Azure CLI
az keyvault secret set \
  --vault-name kv-openjustice-dev \
  --name "gemini-api-key" \
  --value "<your-gemini-api-key>"

az keyvault secret set \
  --vault-name kv-openjustice-dev \
  --name "postgres-password" \
  --value "<generated-password>"

az keyvault secret set \
  --vault-name kv-openjustice-dev \
  --name "jwt-secret" \
  --value "<generated-jwt-secret>"

az keyvault secret set \
  --vault-name kv-openjustice-dev \
  --name "secret-key" \
  --value "<generated-secret-key>"
```

### 5. Connect AKS to ACR

```bash
# Attach ACR to AKS for image pull
az aks update \
  --resource-group rg-openjustice-dev \
  --name aks-openjustice-dev \
  --attach-acr <acr-name>
```

### 6. Get AKS Credentials

```bash
az aks get-credentials \
  --resource-group rg-openjustice-dev \
  --name aks-openjustice-dev \
  --overwrite-existing

# Verify connectivity
kubectl get nodes
```

## Environment Management

### Dev Environment
- Single-node AKS, smaller VM sizes
- Free tier Static Web Apps
- No geo-redundant backups
- 30-day log retention

### Production Environment
- Multi-node AKS with autoscaling (2–10 nodes)
- GPU node pool for model inference
- Zone-redundant PostgreSQL
- Premium Front Door with WAF
- Geo-redundant storage
- 365-day log retention
- 90-day soft-delete Key Vault with purge protection

## Post-Deployment Checklist

- [ ] Deploy K8s manifests: `kubectl apply -f k8s/`
- [ ] Configure ingress DNS with Azure Front Door or Cloudflare
- [ ] Set up cert-manager ClusterIssuer for Let's Encrypt
- [ ] Verify Application Insights data flowing
- [ ] Configure PostgreSQL firewall (or use private endpoint for prod)
- [ ] Set up Azure AD workload identity for AKS → Key Vault access
- [ ] Run integration tests against deployed API
- [ ] Verify monitoring alerts trigger correctly
- [ ] Configure backup jobs (see `infrastructure/backup/`)

## Resource Naming Convention

| Resource | Pattern | Example |
|----------|---------|---------|
| Resource Group | `rg-{app}-{env}` | `rg-openjustice-dev` |
| AKS Cluster | `aks-{app}-{env}` | `aks-openjustice-dev` |
| ACR | `acr{app}{env}` | `acropenjusticedev` |
| PostgreSQL | `psql-{app}-{env}` | `psql-openjustice-dev` |
| Storage Account | `st{app}{env}` | `stopenjusticedev` |
| Key Vault | `kv-{app}-{env}` | `kv-openjustice-dev` |
| Log Analytics | `log-{app}-{env}` | `log-openjustice-dev` |
| App Insights | `appi-{app}-{env}` | `appi-openjustice-dev` |
| Static Web App | `aswa-{app}-{env}` | `aswa-openjustice-dev` |
| Front Door | `afd-{app}` | `afd-openjustice` |

## Cleanup

```bash
# Delete entire resource group (caution: irreversible)
az group delete --name rg-openjustice-dev --yes --no-wait

# Or remove individual resources
az deployment sub delete --name openjustice-dev-deploy
```

## Security

- All secrets stored in Azure Key Vault with RBAC
- AKS uses Managed Identity (no service principals)
- Private endpoints for data services in production
- WAF policies block SQL injection, XSS, and bot traffic
- Network policies enforce pod-level segmentation
- TLS 1.2 minimum on all services
- Container images signed and vulnerability-scanned
