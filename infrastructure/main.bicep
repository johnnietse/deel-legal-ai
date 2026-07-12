// OpenJustice.ai — Lightweight Infrastructure for Azure for Students
// Deploys: PostgreSQL, Blob Storage, Key Vault, Log Analytics, App Insights, Static Web Apps
//
// Usage:
//   az deployment group create \
//     --resource-group rg-openjustice-dev \
//     --template-file infrastructure/main.bicep \
//     --parameters environment=dev

@description('Environment name (dev, staging, prod)')
@allowed(['dev', 'staging', 'prod'])
param environment string = 'dev'

@description('Azure region for deployment')
param location string = 'eastus'

@description('Application name used for resource naming')
param appName string = 'openjustice'

@description('Deploy PostgreSQL Flexible Server (may be restricted on some subscription types)')
param deployPostgres bool = true

@description('PostgreSQL admin password (auto-generated if empty)')
@secure()
param postgresPassword string = uniqueString(resourceGroup().id)

// PostgreSQL Configuration
param postgresBackupRetentionDays int = 7

// Storage Configuration
param storageSku string = 'Standard_LRS'
param storageRetentionDays int = 7

// Monitoring Configuration
param logRetentionDays int = 30  // Minimum for pay-as-you-go

// Resource naming
var resourceSuffix = environment == 'prod' ? '' : '-${environment}'

// Common tags
var commonTags = {
  application: appName
  environment: environment
  managedBy: 'bicep'
}

// ================================
// PostgreSQL Flexible Server (conditional - some sub types restrict this)
// ================================
resource postgres 'Microsoft.DBforPostgreSQL/flexibleServers@2024-08-01' = if (deployPostgres) {
  name: 'psql-${appName}${resourceSuffix}'
  location: location
  sku: {
    name: 'Standard_B1ms'
    tier: 'Burstable'
  }
  properties: {
    administratorLogin: 'openjustice_admin'
    administratorLoginPassword: postgresPassword
    version: '16'
    storage: {
      storageSizeGB: 32
    }
    backup: {
      backupRetentionDays: postgresBackupRetentionDays
      geoRedundantBackup: 'Disabled'
    }
    highAvailability: {
      mode: 'Disabled'
    }
  }
  tags: commonTags
}

// ================================
// Blob Storage
// ================================
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${uniqueString(resourceGroup().id)}'
  location: location
  kind: 'StorageV2'
  sku: {
    name: storageSku
  }
  properties: {
    accessTier: 'Hot'
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: false
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
  tags: commonTags
}

// Storage containers
resource blobServices 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: storageRetentionDays
    }
    containerDeleteRetentionPolicy: {
      enabled: true
      days: storageRetentionDays
    }
  }
}

resource documentsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobServices
  name: 'documents'
}

resource uploadsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobServices
  name: 'uploads'
}

resource backupsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobServices
  name: 'backups'
}

// ================================
// Key Vault
// ================================
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: take('kv-${appName}${uniqueString(resourceGroup().id)}', 24)
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
  tags: commonTags
}

// ================================
// Log Analytics Workspace
// ================================
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${appName}${resourceSuffix}'
  location: location
  properties: {
    retentionInDays: logRetentionDays
    workspaceCapping: {
      dailyQuotaGb: 2
    }
  }
  tags: commonTags
}

// ================================
// Application Insights
// ================================
resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${appName}${resourceSuffix}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    SamplingPercentage: 100
    DisableIpMasking: false
  }
  tags: commonTags
}

// ================================
// Azure Static Web Apps (frontend hosting)
// ================================
resource swa 'Microsoft.Web/staticSites@2022-09-01' = {
  name: 'aswa-${appName}${resourceSuffix}'
  location: 'eastus2'
  properties: {
    repositoryUrl: 'https://github.com/openjustice/openjustice'
    branch: 'main'
    buildProperties: {
      appLocation: 'openjustice-frontend'
      outputLocation: 'dist'
    }
    stagingEnvironmentPolicy: 'Enabled'
  }
  sku: {
    name: 'Free'
    tier: 'Free'
  }
  tags: commonTags
}

// ================================
// Outputs
// ================================
output resourceGroupName string = resourceGroup().name
output postgresDeployed bool = deployPostgres
output postgresServerName string = deployPostgres ? postgres.name : ''
output postgresConnectionString string = deployPostgres ? 'postgresql://openjustice_admin@${postgres.name}:5432/openjustice?sslmode=require' : ''
output storageAccountName string = storage.name
output keyVaultName string = keyVault.name
output keyVaultUri string = keyVault.properties.vaultUri
output logAnalyticsWorkspaceId string = logAnalytics.id
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output swaDefaultHostname string = swa.properties.defaultHostname
