@echo off
setlocal
pushd "%~dp0.."
if not exist ".venv-r4\Scripts\python.exe" (
    py -3.12 -m venv .venv-r4
    if errorlevel 1 goto failed
)
".venv-r4\Scripts\python.exe" -m pip install -r code\business_entity_resolution\requirements_v2.txt
if errorlevel 1 goto failed
".venv-r4\Scripts\python.exe" scripts\run_r4_desktop.py %*
if errorlevel 1 goto failed
popd
exit /b 0
:failed
set "R4_EXIT=%ERRORLEVEL%"
popd
exit /b %R4_EXIT%
