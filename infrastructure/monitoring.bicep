// OpenJustice.ai — Monitoring Configuration
// Deploys: Alert rules, Action Groups, Dashboard
//
// Deploy after main.bicep:
//   az deployment group create \
//     --resource-group rg-openjustice-<env> \
//     --template-file infrastructure/monitoring.bicep \
//     --parameters environment=prod

param environment string = 'dev'
param location string = 'eastus'
param appInsightsConnectionString string = ''
param logAnalyticsWorkspaceId string = ''

// ================================
// Action Groups
// ================================
resource defaultActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: 'ag-openjustice-${environment}-critical'
  location: 'global'
  properties: {
    groupShortName: 'oj-critical'
    enabled: true
    emailReceivers: [
      {
        name: 'oncall-email'
        emailAddress: 'oncall@openjustice.ai'
        useCommonAlertSchema: true
      }
    ]
    smsReceivers: environment == 'prod' ? [
      {
        name: 'oncall-sms'
        countryCode: '1'
        phoneNumber: ''
      }
    ] : []
    webhookReceivers: [
      {
        name: 'pagerduty'
        serviceUri: ''
        useCommonAlertSchema: true
      }
    ]
  }
}

resource warningActionGroup 'Microsoft.Insights/actionGroups@2023-01-01' = {
  name: 'ag-openjustice-${environment}-warning'
  location: 'global'
  properties: {
    groupShortName: 'oj-warning'
    enabled: true
    emailReceivers: [
      {
        name: 'team-email'
        emailAddress: 'team@openjustice.ai'
        useCommonAlertSchema: true
      }
    ]
  }
}

// ================================
// Metric Alert Rules
// ================================

// 1. API High Latency (p95 > 5s for 5 minutes)
resource alertHighLatency 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-openjustice-${environment}-high-latency'
  location: 'global'
  properties: {
    description: 'API p95 latency exceeds 5 seconds for 5 minutes'
    severity: 1
    enabled: true
    scopes: [
      logAnalyticsWorkspaceId
    ]
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    criteria: {
      allOf: [
        {
          criterionType: 'StaticThresholdCriterion'
          metricName: 'requests/duration'
          metricNamespace: 'Microsoft.Insights/Components'
          operator: 'GreaterThan'
          threshold: 5000
          timeAggregation: 'Percentile95'
          skipMetricValidation: true
        }
      ]
    }
    actions: [
      {
        actionGroupId: defaultActionGroup.id
      }
    ]
  }
}

// 2. API Error Rate Spike (>5% errors over 5 minutes)
resource alertErrorRate 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-openjustice-${environment}-error-rate'
  location: 'global'
  properties: {
    description: 'API error rate exceeds 5% for 5 minutes'
    severity: 1
    enabled: true
    scopes: [
      logAnalyticsWorkspaceId
    ]
    evaluationFrequency: 'PT1M'
    windowSize: 'PT5M'
    criteria: {
      allOf: [
        {
          criterionType: 'StaticThresholdCriterion'
          metricName: 'requests/failed'
          metricNamespace: 'Microsoft.Insights/Components'
          operator: 'GreaterThan'
          threshold: 5
          timeAggregation: 'Average'
          skipMetricValidation: true
        }
      ]
    }
    actions: [
      {
        actionGroupId: defaultActionGroup.id
      }
    ]
  }
}

// 3. AKS Node Unavailable
resource alertNodeDown 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-openjustice-${environment}-node-down'
  location: 'global'
  properties: {
    description: 'AKS node count drops below expected minimum'
    severity: 0
    enabled: true
    scopes: [
      logAnalyticsWorkspaceId
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      allOf: [
        {
          criterionType: 'StaticThresholdCriterion'
          metricName: 'kube_node_status_condition'
          metricNamespace: 'Insights.Container'
          operator: 'LessThan'
          threshold: environment == 'prod' ? 2 : 1
          timeAggregation: 'Minimum'
          skipMetricValidation: true
        }
      ]
    }
    actions: [
      {
        actionGroupId: defaultActionGroup.id
      }
    ]
  }
}

// 4. PostgreSQL Connection Limit Near Threshold
resource alertPostgresConnections 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-openjustice-${environment}-pg-connections'
  location: 'global'
  properties: {
    description: 'PostgreSQL active connections exceed 80% of max'
    severity: 2
    enabled: true
    scopes: [
      logAnalyticsWorkspaceId
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      allOf: [
        {
          criterionType: 'StaticThresholdCriterion'
          metricName: 'connections_active'
          metricNamespace: 'Microsoft.DBforPostgreSQL/flexibleServers'
          operator: 'GreaterThan'
          threshold: 80
          timeAggregation: 'Maximum'
          skipMetricValidation: true
        }
      ]
    }
    actions: [
      {
        actionGroupId: warningActionGroup.id
      }
    ]
  }
}

// 5. Disk Space Warning
resource alertDiskSpace 'Microsoft.Insights/metricAlerts@2018-03-01' = {
  name: 'alert-openjustice-${environment}-disk-space'
  location: 'global'
  properties: {
    description: 'Available disk space below 10% on any node'
    severity: 2
    enabled: true
    scopes: [
      logAnalyticsWorkspaceId
    ]
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    criteria: {
      allOf: [
        {
          criterionType: 'StaticThresholdCriterion'
          metricName: 'disk/percentage'
          metricNamespace: 'Insights.Container'
          operator: 'GreaterThan'
          threshold: 90
          timeAggregation: 'Average'
          skipMetricValidation: true
        }
      ]
    }
    actions: [
      {
        actionGroupId: warningActionGroup.id
      }
    ]
  }
}

// 6. RAG Pipeline Latency (App Insights custom metric)
resource alertRagLatency 'Microsoft.Insights/scheduledQueryRules@2023-12-01' = {
  name: 'alert-openjustice-${environment}-rag-latency'
  location: location
  properties: {
    displayName: 'RAG Pipeline Latency Warning'
    description: 'RAG query p99 latency exceeds 30 seconds'
    severity: 2
    enabled: true
    evaluationFrequency: 'PT5M'
    windowSize: 'PT15M'
    scopes: [
      logAnalyticsWorkspaceId
    ]
    criteria: {
      allOf: [
        {
          query: 'requests | where name contains "rag_query" | summarize p99_duration_seconds = percentile(duration, 99) / 1000 by bin(timestamp, 5m) | where p99_duration_seconds > 30'
          timeAggregation: 'Count'
          dimensions: []
          trigger: {
            thresholdOperator: 'GreaterThan'
            threshold: 0
            metricTrigger: {
              metricColumn: 'p99_duration_seconds'
              metricTriggerType: 'Consecutive'
              metricEvaluationInterval: 'PT15M'
              metricEvaluationOffset: 'PT5M'
              operator: 'GreaterThan'
              threshold: 30
              frequency: 'PT5M'
            }
          }
        }
      ]
    }
    actions: {
      actionGroups: [
        warningActionGroup.id
      ]
    }
  }
}

// ================================
// Smart Detection Rules (Application Insights)
// ================================
resource smartDetectionFailure 'Microsoft.Insights/proactiveDetectionConfigs@2018-05-01-preview' = {
  name: 'FailureAnomalies'
  location: 'global'
  properties: {
    enabled: true
    sendEmailsToSubscriptionOwners: false
    customEmails: ['oncall@openjustice.ai']
  }
}

// ================================
// Outputs
// ================================
output defaultActionGroupId string = defaultActionGroup.id
output warningActionGroupId string = warningActionGroup.id
output alertCount int = 6
