@echo off
REM ============================================================
REM OpenJustice.ai CSV Ingestion - Manual Run Script
REM ============================================================
REM Runs the CSV ingestion wrapper with configurable options.
REM Usage: run_ingestion.bat [batch_size] [max_retries] [reset]
REM Example: run_ingestion.bat 20 3
REM Example: Run with batch size 20, max 3 retries
REM          run_ingestion.bat 10 2 reset: Reset progress and run batch of 10
REM ============================================================

setlocal enabledelayedexpansion

REM Default values
set BATCH_SIZE=20
set MAX_RETRIES=3
set RESET_FLAG=

REM Parse arguments
if "%~1" neq "" set BATCH_SIZE=%~1
if "%~2" neq "" set MAX_RETRIES=%~2
if "%~3" equ "reset" set RESET_FLAG=--reset

echo ============================================================
echo OpenJustice.ai CSV Ingestion - Manual Run
echo ============================================================
echo Batch size: %BATCH_SIZE%
echo Max retries: %MAX_RETRIES%
if "%RESET_FLAG%"=="--reset" echo Reset progress: YES
echo Working directory: %~dp0
echo ============================================================

REM Check if Python is available
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please ensure Python 3.11+ is installed and in PATH
    pause
    exit /b 1
)

REM Check if wrapper script exists
if not exist "rag_pipeline\csv_ingestion_wrapper.py" (
    echo ERROR: Wrapper script not found at rag_pipeline\csv_ingestion_wrapper.py
    echo Please run from the project root directory
    pause
    exit /b 1
)

REM Run the wrapper
echo Starting ingestion...
echo.

python rag_pipeline/csv_ingestion_wrapper.py --batch-size %BATCH_SIZE% --max-retries %MAX_RETRIES% %RESET_FLAG%

set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================================
if %EXIT_CODE% equ 0 (
    echo Ingestion completed successfully!
) else (
    echo Ingestion failed with exit code %EXIT_CODE%
    echo Check logs above for details.
)
echo ============================================================

pause
exit /b %EXIT_CODE%