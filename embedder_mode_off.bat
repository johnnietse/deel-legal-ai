@echo off
REM ==========================================
REM Disable Embedder Mode
REM Restores normal sleep/lid close behavior
REM Run this after overnight embedder finishes
REM ==========================================
cd /d "%~dp0"

echo Disabling embedder mode...
echo - Lid close: Sleep
echo - Sleep on AC: 30 minutes
echo - Hibernate: 60 minutes

REM Restore lid close to sleep (index 1 = sleep)
powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS 5ca83367-6e45-459f-a27b-476b1d01c936 1 >nul
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS 5ca83367-6e45-459f-a27b-476b1d01c936 1 >nul

REM Restore sleep timeout (30 min AC, 15 min DC)
powercfg /change standby-timeout-ac 30 >nul
powercfg /change standby-timeout-dc 15 >nul

REM Restore hibernate
powercfg /change hibernate-timeout-ac 60 >nul
powercfg /change hibernate-timeout-dc 30 >nul

powercfg /setactive SCHEME_CURRENT >nul

echo.
echo ✅ Embedder mode OFF - Normal power settings restored.
echo.
echo Laptop will sleep normally when lid is closed.
pause
