param(
    [string]$OutputDirectory,
    [string]$PortableRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$project = Join-Path $repoRoot 'launcher\AI.Toolkit.Launcher\AI.Toolkit.Launcher.csproj'
$cacheRoot = Join-Path $repoRoot '.cache'
$packageCache = Join-Path $cacheRoot 'nuget'
$dotnetHome = Join-Path $cacheRoot 'dotnet-home'

$versionSource = Get-Content -LiteralPath (Join-Path $repoRoot 'version.py') -Raw
$match = [regex]::Match($versionSource, 'VERSION\s*=\s*["''](?<version>[^"'']+)["'']')
if (-not $match.Success) {
    throw 'Could not read VERSION from version.py.'
}
$version = $match.Groups['version'].Value

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $repoRoot "dist\windows-launcher\$version\win-x64"
}
$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)

New-Item -ItemType Directory -Path $cacheRoot, $packageCache, $dotnetHome, $OutputDirectory -Force | Out-Null
$env:DOTNET_CLI_HOME = $dotnetHome
$env:DOTNET_CLI_TELEMETRY_OPTOUT = '1'

$publishArgs = @(
    'publish',
    $project,
    '--configuration', 'Release',
    '--runtime', 'win-x64',
    '--self-contained', 'true',
    '--output', $OutputDirectory,
    '--source', 'https://api.nuget.org/v3/index.json',
    '--packages', $packageCache,
    '-p:PublishSingleFile=true',
    '-p:IncludeNativeLibrariesForSelfExtract=true',
    '-p:EnableCompressionInSingleFile=true',
    '-p:DebugType=None',
    '-p:DebugSymbols=false',
    "-p:Version=$version",
    "-p:FileVersion=$version.0"
)

& dotnet @publishArgs
if ($LASTEXITCODE -ne 0) {
    throw "dotnet publish failed with exit code $LASTEXITCODE."
}

$launcher = Join-Path $OutputDirectory 'AI Toolkit Launcher.exe'
if (-not (Test-Path -LiteralPath $launcher -PathType Leaf)) {
    throw "Published launcher was not found at $launcher."
}

$unexpected = Get-ChildItem -LiteralPath $OutputDirectory -File | Where-Object {
    $_.FullName -ne $launcher
}
if ($unexpected) {
    throw "Single-file publish produced unexpected files: $($unexpected.Name -join ', ')"
}

if (-not [string]::IsNullOrWhiteSpace($PortableRoot)) {
    $PortableRoot = [System.IO.Path]::GetFullPath($PortableRoot)
    if (-not (Test-Path -LiteralPath (Join-Path $PortableRoot 'manager\__main__.py'))) {
        throw "PortableRoot is not an AI Toolkit checkout: $PortableRoot"
    }
    $sourceManager = Join-Path $repoRoot 'manager'
    $targetManager = Join-Path $PortableRoot 'manager'
    Get-ChildItem -LiteralPath $sourceManager -File -Recurse | Where-Object {
        $_.Extension -ne '.pyc' -and $_.FullName -notmatch '[\\/]__pycache__[\\/]'
    } | ForEach-Object {
        $relative = $_.FullName.Substring($sourceManager.Length).TrimStart('\', '/')
        $target = Join-Path $targetManager $relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
    }
    Copy-Item -LiteralPath $launcher -Destination (Join-Path $PortableRoot 'AI Toolkit Launcher.exe') -Force
    Write-Host "Matching manager and launcher copied to $PortableRoot"
}

$file = Get-Item -LiteralPath $launcher
Write-Host "Published $($file.FullName) ($([math]::Round($file.Length / 1MB, 1)) MB)"
