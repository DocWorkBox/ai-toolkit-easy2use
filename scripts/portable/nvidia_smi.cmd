@echo off
setlocal EnableExtensions

if defined NVIDIA_SMI_PATH if exist "%NVIDIA_SMI_PATH%" goto :run

if defined ProgramW6432 if exist "%ProgramW6432%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
  set "NVIDIA_SMI_PATH=%ProgramW6432%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
  goto :run
)

if defined ProgramFiles if exist "%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe" (
  set "NVIDIA_SMI_PATH=%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe"
  goto :run
)

if defined SystemRoot if exist "%SystemRoot%\System32\nvidia-smi.exe" (
  set "NVIDIA_SMI_PATH=%SystemRoot%\System32\nvidia-smi.exe"
  goto :run
)

for /f "delims=" %%I in ('where nvidia-smi 2^>nul') do (
  if not defined NVIDIA_SMI_PATH set "NVIDIA_SMI_PATH=%%~fI"
)

if not defined NVIDIA_SMI_PATH (
  echo nvidia-smi not found or not accessible 1>&2
  exit /b 1
)

:run
"%NVIDIA_SMI_PATH%" %*
