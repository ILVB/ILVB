# Clean-install test (Windows PowerShell): fresh clone -> new venv -> install -> doctor -> demo.
# Usage: powershell -ExecutionPolicy Bypass -File scripts\clean_install_test.ps1 [REPO]
param([string]$Repo = (git rev-parse --show-toplevel), [string]$Extras = "rapid,gui")
$ErrorActionPreference = "Stop"
$Work = Join-Path ([IO.Path]::GetTempPath()) ("mangaar-clean-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Work | Out-Null
Write-Host "== clone $Repo -> $Work\src"
git clone --quiet $Repo "$Work\src"
Set-Location "$Work\src"
$env:MANGAAR_CACHE_DIR = "$Work\cache"
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv venv --quiet --python 3.11 .venv
    $args = @(); foreach ($e in $Extras.Split(",")) { $args += @("--extra", $e.Trim()) }
    uv sync --quiet --locked @args
} else {
    python -m venv .venv
    .venv\Scripts\pip install --quiet -e ".[$Extras]"
    .venv\Scripts\pip uninstall --quiet -y opencv-python opencv-contrib-python
    .venv\Scripts\pip install --quiet --force-reinstall opencv-python-headless
}
.venv\Scripts\manga-arabic --version
.venv\Scripts\manga-arabic doctor --no-network
if ($LASTEXITCODE -ne 0) { throw "doctor failed" }
.venv\Scripts\manga-arabic demo -o "$Work\demo"
if ($LASTEXITCODE -ne 0) { throw "demo failed" }
if (-not (Test-Path "$Work\demo\output\demo_zh_ar.png")) { throw "no demo output" }
Write-Host "CLEAN INSTALL OK ($Work)"
