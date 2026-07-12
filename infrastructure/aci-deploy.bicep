// OpenJustice.ai - Azure Container Instance deployment
param acrPassword string
param location string = 'eastus2'

resource aci 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: 'aci-openjustice-api'
  location: location
  properties: {
    containers: [
      {
        name: 'openjustice-api'
        properties: {
          image: 'acropenjustice.azurecr.io/openjustice-api:latest'
          ports: [
            {
              port: 8000
              protocol: 'TCP'
            }
          ]
          resources: {
            requests: {
              cpu: 1
              memoryInGB: 1.5
            }
          }
          environmentVariables: [
            { name: 'DEV_MODE', value: '0' }
            { name: 'GEMINI_API_KEY', secureValue: 'AQ.Ab8RN6LAwOt_g3lTL_EUwQptXyWdM2FLJdAhLknMmc54nvI8XA' }
            { name: 'PINECONE_API_KEY', secureValue: 'pcsk_5rfJUm_FvMtLsRyVxph343zYRaMaSe6QLXDRqPpRbqjP5jQAoaeDnsZjcitns4bMZUwm3Z' }
            { name: 'JWT_SECRET_KEY', secureValue: 'dev-super-secret-jwt-key-change-in-production-min-32-chars' }
          ]
        }
      }
    ]
    imageRegistryCredentials: [
      {
        server: 'acropenjustice.azurecr.io'
        username: 'acropenjustice'
        password: acrPassword
      }
    ]
    restartPolicy: 'Always'
    osType: 'Linux'
    ipAddress: {
      type: 'Public'
      dnsNameLabel: 'openjustice-api'
      ports: [
        {
          port: 8000
          protocol: 'TCP'
        }
      ]
    }
  }
}
