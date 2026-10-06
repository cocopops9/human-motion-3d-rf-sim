@echo off
rem Run a bodyscan command from a Windows command prompt, from any folder:
rem     C:\path\to\lidar-body-scan\bodyscan.bat fuse C:\lidar\tt17 --out C:\lidar\person_tt17
rem Uses the Python found on PATH; set BODYSCAN_PYTHON to use another one, e.g. MATLAB's:
rem     set BODYSCAN_PYTHON=C:\Users\me\AppData\Local\Programs\Python\Python311\python.exe
setlocal
set "REPOSITORY=%~dp0"
if "%BODYSCAN_PYTHON%"=="" set "BODYSCAN_PYTHON=python"
set "PYTHONPATH=%REPOSITORY%;%PYTHONPATH%"
"%BODYSCAN_PYTHON%" -m bodyscan %*
endlocal
