@echo off
REM ==========================================
REM Overnight A2AJ Embedder — Double-click to run
REM Will process employment law cases with
REM 12 Gemini API keys and auto-rotation
REM Keys loaded from .env automatically
REM ==========================================

cd /d "%~dp0"

echo %DATE% %TIME% - Starting overnight A2AJ embedder...
echo API keys: 12 (auto-rotation on rate limits)
echo Output log: data\background_embedder.log
echo.
echo Will process up to 25,000 employment law cases.
echo Close this window to stop.
echo.

"C:\Users\Johnnie\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none\python.exe" rag_pipeline/background_a2aj_embedder.py --max 25000 --delay 1.5 --resume

echo.
echo %DATE% %TIME% - Embedder finished.
echo Press any key to close.
pause
