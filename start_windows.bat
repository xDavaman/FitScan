@echo off
rem FitScan Windows-Prototyp: Doppelklick genügt. Beim ersten Start wird alles eingerichtet
rem (Python-Umgebung, PyTorch mit CUDA 12.8 für RTX 50xx, KI-Modell). Danach läuft FitScan offline.
rem Mit Befehl starten, z. B.:  start_windows.bat live   oder   start_windows.bat selbsttest
chcp 65001 > nul
setlocal
cd /d "%~dp0"
title FitScan

where uv > nul 2> nul
if errorlevel 1 (
    echo [1/4] uv wird installiert ...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex" || goto fehler
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
)

if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Python-Umgebung wird angelegt ...
    uv venv -p 3.12 .venv || goto fehler
)
set "PY=.venv\Scripts\python.exe"

"%PY%" -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" > nul 2> nul
if errorlevel 1 (
    echo [3/4] PyTorch mit CUDA 12.8 wird installiert, beim ersten Mal dauert das einige Minuten ...
    uv pip install -p "%PY%" torch --index-url https://download.pytorch.org/whl/cu128 || goto fehler
)

echo [4/4] FitScan wird eingerichtet oder aktualisiert ...
uv pip install -q -p "%PY%" -e ".[prototyp]" || goto fehler
"%PY%" -c "from fitscan.vision import ensure_pose_model; ensure_pose_model()" || goto fehler
"%PY%" -c "from fitscan.head import ensure_face_model; ensure_face_model()" || goto fehler
"%PY%" -c "from fitscan.hands import ensure_hand_model; ensure_hand_model()" || goto fehler

"%PY%" -m fitscan info
echo.
if "%~1"=="" (
    echo Die Oberfläche öffnet sich im Browser. Dieses Fenster offen lassen, zum Beenden schließen.
)
"%PY%" -m fitscan %*
if errorlevel 1 goto fehler
goto ende

:fehler
echo.
echo Es ist ein Fehler aufgetreten. Bitte die Meldungen oben kopieren und an Claude schicken.
pause
exit /b 1

:ende
endlocal
