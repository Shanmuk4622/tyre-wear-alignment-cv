@echo off
cd /d "%~dp0"
call conda activate cv_conda
if errorlevel 1 goto failed
python app.py %*
if errorlevel 1 goto failed
exit /b 0
:failed
echo Tread Station could not start. Read the message above and README.md.
pause
