@echo off
setlocal
cd /d "%~dp0"
if exist "..\run_cv.cmd" (
  call "..\run_cv.cmd" -B "%~dp0app.py" --example %*
) else (
  call conda activate cv_conda
  if errorlevel 1 goto failed
  python -B "%~dp0app.py" --example %*
)
if errorlevel 1 goto failed
exit /b 0
:failed
echo TREAD-ONE Studio could not start. See README.md and .cache\ui-errors.log.
pause
