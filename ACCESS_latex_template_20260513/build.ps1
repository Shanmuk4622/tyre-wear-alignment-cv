param([switch]$RefreshFigures)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if ($RefreshFigures) {
        python build_assets.py
        if ($LASTEXITCODE -ne 0) { throw 'Evidence/figure build failed' }
        python revision_analysis.py
        if ($LASTEXITCODE -ne 0) { throw 'Revision analysis failed' }
    }
    $paperCompiler = Get-Command tectonic -ErrorAction SilentlyContinue
    $localCompiler = Join-Path $PSScriptRoot '../prototype/.cache/latex-tools/tectonic.exe'
    if ($paperCompiler -or (Test-Path -LiteralPath $localCompiler)) {
        $env:TECTONIC_CACHE_DIR = Join-Path $PSScriptRoot '../prototype/.cache/tectonic'
        $compilerPath = if ($paperCompiler) { $paperCompiler.Source } else { $localCompiler }
        & $compilerPath -X compile main.tex --keep-logs --keep-intermediates
        if ($LASTEXITCODE -ne 0) { throw 'LaTeX compilation failed; do not use a stale PDF' }
    } elseif (Get-Command pdflatex -ErrorAction SilentlyContinue) {
        pdflatex -interaction=nonstopmode -halt-on-error main.tex
        if ($LASTEXITCODE -ne 0) { throw 'First LaTeX pass failed' }
        bibtex main
        if ($LASTEXITCODE -ne 0) { throw 'Bibliography build failed' }
        1..2 | ForEach-Object {
            pdflatex -interaction=nonstopmode -halt-on-error main.tex
            if ($LASTEXITCODE -ne 0) { throw 'LaTeX reference pass failed' }
        }
    } else {
        throw 'Use the included sources in Overleaf (main.tex, pdfLaTeX), or install Tectonic/pdfLaTeX locally.'
    }
    python verify_paper.py
    if ($LASTEXITCODE -ne 0) { throw 'Paper verification failed' }
} finally {
    Pop-Location
}
