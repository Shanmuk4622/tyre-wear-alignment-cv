@echo off
rem Use Conda's Python entry point when the installed conda.exe wrapper fails.
set "CONDA_EXE=C:\Users\shanm\dev\python.exe"
set "_CE_M=-m"
set "_CE_CONDA=conda"
call C:\Users\shanm\dev\condabin\conda.bat activate cv_conda
if errorlevel 1 exit /b %errorlevel%
python %*
exit /b %errorlevel%
