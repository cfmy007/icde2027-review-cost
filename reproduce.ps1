# Recompute the paper's tables from the stored measurements (Windows PowerShell).
$ErrorActionPreference = "Stop"
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = "python" }
& $py reproduce.py
exit $LASTEXITCODE
