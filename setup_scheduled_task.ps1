<#>
.SYNOPSIS
    Sets up Windows Task Scheduler for automated CSV ingestion.

.DESCRIPTION
    Creates a scheduled task that runs the CSV ingestion wrapper every 30 minutes.
    The wrapper handles rate limits, tracks progress, and resumes automatically.

.PARAMETER TaskName
    Name of the scheduled task (default: "OpenJustice-CSV-Ingestion")

.PARAMETER BatchSize
    Number of cases per run (default: 20)

.PARAMETER MaxRetries
    Max retries per batch on quota exhaustion (default: 3)

.PARAMETER PythonPath
    Path to python.exe (auto-detected if not provided)

.PARAMETER WorkingDirectory
    Working directory for the task (default: script directory)

.EXAMPLE
    .\setup_scheduled_task.ps1 -BatchSize 20 -MaxRetries 3

.EXAMPLE
    .\setup_scheduled_task.ps1 -TaskName "My-Ingestion" -BatchSize 10 -MaxRetries 5
#>

param(
    [string]$TaskName = "OpenJustice-CSV-Ingestion",
    [int]$BatchSize = 20,
    [int]$MaxRetries = 3,
    [string]$PythonPath = "",
    [string]$WorkingDirectory = ""
)

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
if (-not $WorkingDirectory) { $WorkingDirectory = $ScriptDir }

# Auto-detect Python path
if (-not $PythonPath) {
    $PythonPath = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $PythonPath) {
        $PythonPath = "C:\Users\$env:USERNAME\AppData\Local\hermes\hermes-agent\venv\python.exe"
    }
}

# Verify paths
$WrapperScript = Join-Path $WorkingDirectory "rag_pipeline\csv_ingestion_wrapper.py"
if (-not (Test-Path $WrapperScript)) {
    Write-Error "Wrapper script not found: $WrapperScript"
    exit 1
}

if (-not (Test-Path $PythonPath)) {
    Write-Error "Python not found at: $PythonPath"
    exit 1
}

# Build arguments
$Arguments = "-File `"$WrapperScript`" --batch-size $BatchSize --max-retries $MaxRetries"

# Create action
$Action = New-ScheduledTaskAction -Execute $PythonPath -Argument $Arguments -WorkingDirectory $WorkingDirectory

# Create trigger (every 30 minutes)
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 30) -RepetitionDuration ([TimeSpan]::FromDays(3650))

# Create settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Register task
try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $Action `
        -Trigger $Trigger `
        -Settings $Settings `
        -Description "Automated CSV ingestion for OpenJustice.ai vector database. Runs every 30 minutes, processes $BatchSize cases per run with $MaxRetries retries on quota exhaustion." `
        -Force `
        -ErrorAction Stop
    
    Write-Host "✅ Scheduled task '$TaskName' created successfully!" -ForegroundColor Green
    Write-Host "   Runs every 30 minutes" -ForegroundColor Cyan
    Write-Host "   Batch size: $BatchSize cases per run" -ForegroundColor Cyan
    Write-Host "   Max retries: $MaxRetries" -ForegroundColor Cyan
    Write-Host "   Working directory: $WorkingDirectory" -ForegroundColor Cyan
    Write-Host "   Python: $PythonPath" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To view task: Task Scheduler > Task Scheduler Library > $TaskName" -ForegroundColor Yellow
    Write-Host "To run manually: Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host "To disable: Disable-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Yellow
    Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false" -ForegroundColor Yellow
}
catch {
    Write-Error "Failed to create scheduled task: $_"
    exit 1
}