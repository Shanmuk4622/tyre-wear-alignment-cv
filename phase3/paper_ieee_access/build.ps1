param([switch]$Assets)
$ErrorActionPreference = 'Stop'
$paperRoot = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $paperRoot '../..')).Path
if ($Assets) {
    Push-Location $repoRoot
    try {
        & (Join-Path $repoRoot 'phase3/run_cv.cmd') -B (Join-Path $paperRoot 'build_assets.py')
        if ($LASTEXITCODE -ne 0) { throw 'Figure build failed' }
    } finally { Pop-Location }
}
$tectonicPath = Join-Path $repoRoot 'prototype/.cache/latex-tools/tectonic.exe'
if (-not (Test-Path -LiteralPath $tectonicPath)) {
    $tectonicCommand = Get-Command tectonic -ErrorAction SilentlyContinue
    if ($tectonicCommand) { $tectonicPath = $tectonicCommand.Source }
    else { throw 'Tectonic is not available. Use a full TeX installation or import this source folder into Overleaf.' }
}
$paperCache = Join-Path $paperRoot '.build/cache'
New-Item -ItemType Directory -Force -Path $paperCache | Out-Null
$previousCache = $env:TECTONIC_CACHE_DIR
$env:TECTONIC_CACHE_DIR = $paperCache
Push-Location $paperRoot
try {
    & $tectonicPath -X compile main.tex --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed; inspect main.log.' }
} finally {
    Pop-Location
    $env:TECTONIC_CACHE_DIR = $previousCache
}
