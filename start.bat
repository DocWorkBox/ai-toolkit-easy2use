@echo off
setlocal EnableExtensions

if exist "%~dp0runtime\python\python.exe" (
  for %%I in ("%~dp0.") do set "ROOT=%%~fI"
) else (
  for %%I in ("%~dp0..\..") do set "ROOT=%%~fI"
)

set "PYTHON_RUNTIME=%ROOT%\runtime\python"
set "NODE_RUNTIME=%ROOT%\runtime\node"
set "FFMPEG_RUNTIME=%ROOT%\runtime\ffmpeg\bin"
set "NODE_EXE=%NODE_RUNTIME%\node.exe"
set "UI_DIR=%ROOT%\ui"
set "PRISMA_JS=%ROOT%\ui\node_modules\prisma\build\index.js"
set "FILE_SERVER_JS=%ROOT%\ui\dist\cron\fileServer.js"
set "NVIDIA_SMI_HELPER=%ROOT%\scripts\portable\nvidia_smi.cmd"
set "SUPERVISOR_PS1=%ROOT%\scripts\portable\run_portable_supervisor.ps1"
set "APP_PORT=8675"
set "APP_URL=http://127.0.0.1:8675"
set "CACHE_DIR=%ROOT%\.cache"
set "WORKER_OUT=%CACHE_DIR%\worker.stdout.log"
set "WORKER_ERR=%CACHE_DIR%\worker.stderr.log"
set "UI_OUT=%CACHE_DIR%\ui.stdout.log"
set "UI_ERR=%CACHE_DIR%\ui.stderr.log"
set "PORT_OWNER_INFO=%CACHE_DIR%\port_owner.txt"

if not exist "%PYTHON_RUNTIME%\python.exe" call :fail "[portable] Missing bundled Python runtime: %PYTHON_RUNTIME%"
if not exist "%NODE_EXE%" call :fail "[portable] Missing bundled Node runtime: %NODE_EXE%"
if not exist "%PRISMA_JS%" call :fail "[portable] Missing Prisma CLI entry: %PRISMA_JS%"
if not exist "%FILE_SERVER_JS%" call :fail "[portable] Missing file server entry: %FILE_SERVER_JS%"
if not exist "%ROOT%\ui\.next" call :fail "[portable] Missing built UI assets: %ROOT%\ui\.next"
if not exist "%NVIDIA_SMI_HELPER%" call :fail "[portable] Missing NVIDIA helper: %NVIDIA_SMI_HELPER%"
if not exist "%SUPERVISOR_PS1%" call :fail "[portable] Missing supervisor script: %SUPERVISOR_PS1%"
if exist "%FFMPEG_RUNTIME%\ffmpeg.exe" (
  set "PATH=%FFMPEG_RUNTIME%;%NODE_RUNTIME%;%PYTHON_RUNTIME%;%SystemRoot%\System32;%PATH%"
) else (
  set "PATH=%NODE_RUNTIME%;%PYTHON_RUNTIME%;%SystemRoot%\System32;%PATH%"
)
set "PYTHONHOME=%PYTHON_RUNTIME%"
set "PYTHONPATH=%ROOT%;%PYTHON_RUNTIME%\Lib;%PYTHON_RUNTIME%\Lib\site-packages"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "NPM_CONFIG_CACHE=%CACHE_DIR%\npm"
set "AITK_NVIDIA_SMI_COMMAND=%NVIDIA_SMI_HELPER%"
if not exist "%CACHE_DIR%" mkdir "%CACHE_DIR%" >nul 2>nul
if not exist "%CACHE_DIR%\npm" mkdir "%CACHE_DIR%\npm" >nul 2>nul
if exist "%PORT_OWNER_INFO%" del /q "%PORT_OWNER_INFO%" >nul 2>nul

powershell.exe -NoProfile -Command ^
  "$connection = Get-NetTCPConnection -LocalPort %APP_PORT% -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1;" ^
  "if (-not $connection) { exit 1 }" ^
  "$process = Get-CimInstance Win32_Process -Filter ('ProcessId = ' + $connection.OwningProcess);" ^
  "$commandLine = if ($process.CommandLine) { $process.CommandLine } else { '' };" ^
  "Set-Content -Path '%PORT_OWNER_INFO%' -Value $commandLine;" ^
  "if ($commandLine -like '*%NODE_EXE%*' -and $commandLine -like '*%FILE_SERVER_JS%*') { exit 0 } else { exit 2 }" >nul 2>nul
if not errorlevel 1 (
  if errorlevel 2 (
    echo [portable] Port %APP_PORT% is already in use by another process.
    if exist "%PORT_OWNER_INFO%" type "%PORT_OWNER_INFO%"
    echo.
    echo [portable] Please close the existing service, then launch this portable bundle again.
    echo.
    pause
    exit /b 1
  ) else (
    echo [portable] This portable bundle is already running. Opening browser...
    start "" "%APP_URL%"
    exit /b 0
  )
)

if not exist "%ROOT%\aitk_db.db" (
  echo [portable] Initializing SQLite database...
  pushd "%UI_DIR%" >nul
  call "%NODE_EXE%" "%PRISMA_JS%" generate
  if errorlevel 1 (
    popd >nul
    call :fail "[portable] Database initialization failed during prisma generate."
  )
  call "%NODE_EXE%" "%PRISMA_JS%" db push
  if errorlevel 1 (
    popd >nul
    call :fail "[portable] Database initialization failed during prisma db push."
  )
  popd >nul
)

echo [portable] Starting ai-toolkit-easy2use...
call powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%SUPERVISOR_PS1%" -NodeExe "%NODE_EXE%" -UiDir "%UI_DIR%" -FileServerJs "%FILE_SERVER_JS%" -AppPort "%APP_PORT%" -AppUrl "%APP_URL%" -CacheDir "%CACHE_DIR%" -WorkerOut "%WORKER_OUT%" -WorkerErr "%WORKER_ERR%" -UiOut "%UI_OUT%" -UiErr "%UI_ERR%"
exit /b %errorlevel%
:fail
echo %~1
echo.
pause
exit /b 1
