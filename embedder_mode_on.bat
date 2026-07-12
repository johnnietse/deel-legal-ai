@echo off
REM ==========================================
REM Enable Embedder Mode
REM Prevents sleep when lid is closed or plugged in
REM Run this when running overnight embedder
REM ==========================================
cd /d "%~dp0"

echo Enabling embedder mode...
echo - Lid close: Do nothing
echo - Sleep on AC: Never
echo - Hibernate: Disabled

powercfg /setacvalueindex SCHEME_CURRENT SUB_BUTTONS 5ca83367-6e45-459f-a27b-476b1d01c936 0 >nul
powercfg /setdcvalueindex SCHEME_CURRENT SUB_BUTTONS 5ca83367-6e45-459f-a27b-476b1d01c936 0 >nul
powercfg /change standby-timeout-ac 0 >nul
powercfg /change hibernate-timeout-ac 0 >nul
powercfg /setactive SCHEME_CURRENT >nul

echo.
echo ✅ Embedder mode ON - Laptop will stay awake with lid closed.
echo.
echo Run embedder_mode_off.bat to restore normal power settings.
pause
