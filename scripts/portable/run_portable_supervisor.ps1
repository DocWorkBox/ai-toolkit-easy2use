param(
    [Parameter(Mandatory = $true)]
    [string]$NodeExe,
    [Parameter(Mandatory = $true)]
    [string]$UiDir,
    [Parameter(Mandatory = $true)]
    [string]$FileServerJs,
    [Parameter(Mandatory = $true)]
    [string]$AppPort,
    [Parameter(Mandatory = $true)]
    [string]$AppUrl,
    [Parameter(Mandatory = $true)]
    [string]$CacheDir,
    [Parameter(Mandatory = $true)]
    [string]$WorkerOut,
    [Parameter(Mandatory = $true)]
    [string]$WorkerErr,
    [Parameter(Mandatory = $true)]
    [string]$UiOut,
    [Parameter(Mandatory = $true)]
    [string]$UiErr,
    [string]$PathValue = $env:PATH,
    [string]$PythonHome = $env:PYTHONHOME,
    [string]$PythonPath = $env:PYTHONPATH,
    [string]$AitkNvidiaSmiCommand = $env:AITK_NVIDIA_SMI_COMMAND
)

$ErrorActionPreference = 'Stop'

function Test-PortListening {
    param([int]$Port)
    return [bool](Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}

function Stop-ChildProcessTree {
    param([System.Diagnostics.Process]$Process)
    if (-not $Process) {
        return
    }
    try {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F 2>$null | Out-Null
    } catch {
    }
}

function Read-LastLogLine {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return ''
    }
    $lines = Get-Content -LiteralPath $Path -ErrorAction SilentlyContinue
    if (-not $lines) {
        return ''
    }
    return $lines[-1]
}

if (-not (Test-Path -LiteralPath $CacheDir)) {
    New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null
}

Remove-Item $WorkerOut, $WorkerErr, $UiOut, $UiErr -Force -ErrorAction SilentlyContinue

$env:PATH = $PathValue
$env:PYTHONHOME = $PythonHome
$env:PYTHONPATH = $PythonPath
$env:AITK_NVIDIA_SMI_COMMAND = $AitkNvidiaSmiCommand

$workerProcess = $null
$uiProcess = $null
$browserOpened = $false
$endpointFile = Join-Path ([IO.Path]::GetTempPath()) ("aitk-ui-{0}.port" -f [guid]::NewGuid().ToString('N'))

try {
    $workerProcess = Start-Process -FilePath $NodeExe `
        -ArgumentList 'dist/cron/worker.js' `
        -WorkingDirectory $UiDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $WorkerOut `
        -RedirectStandardError $WorkerErr `
        -PassThru

    $env:AITK_UI_ENDPOINT_FILE = $endpointFile
    $uiProcess = Start-Process -FilePath $NodeExe `
        -ArgumentList "`"$FileServerJs`" start --port $AppPort" `
        -WorkingDirectory $UiDir `
        -WindowStyle Hidden `
        -RedirectStandardOutput $UiOut `
        -RedirectStandardError $UiErr `
        -PassThru

    $wait_for_port = 0
    while ($wait_for_port -lt 60) {
        if ($uiProcess.HasExited) {
            throw "UI process exited before port $AppPort was ready. $(Read-LastLogLine -Path $UiErr)"
        }
        if (Test-Path -LiteralPath $endpointFile) {
            $bound = [int]([IO.File]::ReadAllText($endpointFile))
            if ($bound -lt 1 -or $bound -gt 65535) { throw 'Invalid UI port' }
            $AppPort = "$bound"
            $AppUrl = "http://127.0.0.1:$bound/"
            try {
                $response = Invoke-WebRequest -Uri $AppUrl -UseBasicParsing -TimeoutSec 2
                if ($response.StatusCode -eq 200) { break }
            } catch { }
        }
        Start-Sleep -Seconds 1
        $wait_for_port++
    }

    if ($wait_for_port -ge 60) {
        throw "Timed out waiting for port $AppPort."
    }

    Start-Process $AppUrl | Out-Null
    $browserOpened = $true
    Write-Host "[portable] UI is ready at $AppUrl"
    Write-Host "[portable] Supervisor is running. If port $AppPort stops listening, this process will exit."

    while ($true) {
        Start-Sleep -Seconds 2
        $monitor_loop = $true

        if ($uiProcess.HasExited) {
            Write-Host "[portable] UI process exited. Shutting down..."
            break
        }

        if (-not (Test-PortListening -Port ([int]$AppPort))) {
            Write-Host "[portable] Port $AppPort is no longer listening. Shutting down..."
            break
        }
    }
} finally {
    Stop-ChildProcessTree -Process $uiProcess
    Stop-ChildProcessTree -Process $workerProcess
    if (Test-Path -LiteralPath $endpointFile) { Remove-Item -LiteralPath $endpointFile }
}
