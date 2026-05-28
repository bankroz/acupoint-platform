@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   MediaPipe 穴位定位 - 最小验证环境
echo ============================================
echo.
echo GPU状态:
echo.

python scripts/demo_verify.py

echo.
pause
