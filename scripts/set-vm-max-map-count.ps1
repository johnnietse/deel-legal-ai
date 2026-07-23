<#
.SYNOPSIS
    Sets vm.max_map_count on WSL2 for Elasticsearch compatibility on Windows.
.DESCRIPTION
    Elasticsearch requires vm.max_map_count >= 262144 on the host kernel.
    On Windows with WSL2-backed Docker, this must be set inside the Docker
    VM via `wsl -d docker-desktop`.
    
    Two approaches are documented below:
      1. One-time fix: Run this script after Docker Desktop starts.
      2. Persistent fix: Configure .wslconfig (see NOTES).
.NOTES
    Persistent approach — add to %USERPROFILE%\.wslconfig:
    
        [wsl2]
        kernelCommandLine = sysctl.vm.max_map_count=262144
    
    Then restart WSL: wsl --shutdown
    
    If you have other WSL distros, you may need to run:
        wsl -d <distro> -u root sysctl -w vm.max_map_count=1048576
.LINK
    https://www.elastic.co/guide/en/elasticsearch/reference/current/vm-max-map-count.html
#>

$ErrorActionPreference = "Stop"

Write-Host "=== Setting vm.max_map_count for Elasticsearch on WSL2 ===" -ForegroundColor Cyan

# Check if Docker Desktop WSL backend is available
$dockerWsl = "docker-desktop"
$wslCheck = wsl -l -q 2>$null | Select-String -Pattern $dockerWsl -SimpleMatch

if (-not $wslCheck) {
    Write-Warning "Docker Desktop WSL VM '$dockerWsl' not found."
    Write-Host "Trying to set on default WSL distro..." -ForegroundColor Yellow
    
    # Try setting on any available WSL distro
    $distros = wsl -l -q 2>$null
    if (-not $distros) {
        Write-Error "No WSL distros found. Install WSL2 and Docker Desktop first."
        exit 1
    }
    
    foreach ($distro in $distros) {
        $distro = $distro.Trim()
        if ($distro -and $distro -ne "docker-desktop-data") {
            Write-Host "Trying distro: $distro" -ForegroundColor Yellow
            wsl -d $distro -u root sysctl -w vm.max_map_count=262144 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "SUCCESS: Set vm.max_map_count on $distro" -ForegroundColor Green
                exit 0
            }
        }
    }
    
    Write-Error "Could not set vm.max_map_count on any WSL distro."
    exit 1
}

# Set on Docker Desktop WSL VM
Write-Host "Setting vm.max_map_count=262144 on $dockerWsl ..." -ForegroundColor Yellow
wsl -d $dockerWsl -u root sysctl -w vm.max_map_count=262144 2>&1

if ($LASTEXITCODE -eq 0) {
    Write-Host "SUCCESS: vm.max_map_count set to 262144" -ForegroundColor Green
    Write-Host "Verifying..." -ForegroundColor Cyan
    wsl -d $dockerWsl -u root sysctl -n vm.max_map_count
} else {
    Write-Error "Failed to set vm.max_map_count. Try running as Administrator."
    Write-Host @"
Alternative: Add to %USERPROFILE%\.wslconfig:
    [wsl2]
    kernelCommandLine = sysctl.vm.max_map_count=262144
Then run: wsl --shutdown && docker-desktop start
"@ -ForegroundColor Yellow
    exit 1
}

# Also stop orphan containers to avoid port conflicts
Write-Host "`nCleaning up orphan containers to prevent port conflicts..." -ForegroundColor Cyan
docker-compose -f "$PSScriptRoot\..\docker-compose.yml" down --remove-orphans 2>$null

Write-Host "`n=== Done. You can now run: docker-compose up -d ===" -ForegroundColor Green
