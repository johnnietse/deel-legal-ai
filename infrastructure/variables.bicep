// OpenJustice.ai — Bicep Variables
// Environment-specific configuration parameters

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Azure region for deployment')
param location string = 'eastus'

@description('Application name used for resource naming')
param appName string = 'openjustice'

// ================================
// Resource Naming
// ================================
var resourceSuffix = environment == 'prod' ? '' : '-${environment}'
var resourceGroupName = 'rg-${appName}${resourceSuffix}'
var aksClusterName = 'aks-${appName}${resourceSuffix}'
var acrName = replace('acr${appName}${resourceSuffix}', '-', '')
var postgresServerName = 'psql-${appName}${resourceSuffix}'
var storageAccountName = replace('st${appName}${resourceSuffix}', '-', '')
var logAnalyticsName = 'log-${appName}${resourceSuffix}'
var appInsightsName = 'appi-${appName}${resourceSuffix}'
var keyVaultName = 'kv-${appName}${resourceSuffix}'
var frontDoorName = 'afd-${appName}${resourceSuffix}'
var swaName = 'aswa-${appName}${resourceSuffix}'

// ================================
// AKS Configuration
// ================================
@description('Kubernetes version for AKS')
param kubernetesVersion string = '1.30'

@description('VM size for AKS node pool')
param aksNodeSize string = 'Standard_D4s_v5'

@description('Minimum node count for AKS cluster')
param aksMinNodes int = 2

@description('Maximum node count for AKS cluster')
param aksMaxNodes int = 10

@description('Enable GPU node pool')
param enableGpuPool bool = false

@description('GPU VM size (when GPU pool enabled)')
param gpuNodeSize string = 'Standard_NC6s_v3'

// ================================
// PostgreSQL Configuration
// ================================
@description('PostgreSQL SKU')
param postgresSku string = 'GP_Standard_D2s_v3'

@description('PostgreSQL storage size in GB')
param postgresStorageGB int = 100

@description('PostgreSQL backup retention days')
param postgresBackupRetentionDays int = 30

@description('PostgreSQL geo-redundant backup')
param postgresGeoRedundantBackup bool = environment == 'prod'

// ================================
// Storage Configuration
// ================================
@description('Blob storage SKU')
param storageSku string = 'Standard_GRS'

@description('Storage blob retention in days')
param storageRetentionDays int = environment == 'prod' ? 365 : 30

// ================================
// Monitoring Configuration
// ================================
@description('Log Analytics retention in days')
param logRetentionDays int = environment == 'prod' ? 365 : 30

@description('Application Insights sampling percentage')
param appInsightsSamplingPercentage int = environment == 'prod' ? 10 : 100

// ================================
// Tags
// ================================
var commonTags = {
  application: appName
  environment: environment
  managedBy: 'bicep'
  costCenter: 'legal-ai'
}

// ================================
// Networking
// ================================
@description('AKS service CIDR range')
param aksServiceCidr string = '10.0.0.0/16'

@description('AKS DNS service IP')
param aksDnsServiceIp string = '10.0.0.10'

@description('AKS Docker bridge CIDR')
param aksDockerBridgeCidr string = '172.17.0.1/16'

@description('Authorized IP ranges for PostgreSQL firewall')
param postgresFirewallRules array = []
