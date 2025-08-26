@echo off
cd /d "%~dp0"
echo =======================================
echo   Iniciando DefectDetect en :7860
echo =======================================

:: Activar el entorno virtual
call .venv\Scripts\activate

:: (Opcional) resetear dataset y modelo en cada arranque
:: python reset_dataset.py

:: Arrancar el servidor
python app.py

pause
