$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$api = Start-Process -FilePath "python" `
    -ArgumentList @("-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", "8000") `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

$web = Start-Process -FilePath "npm.cmd" `
    -ArgumentList @("run", "dev", "--prefix", "apps/web") `
    -WorkingDirectory $root -WindowStyle Hidden -PassThru

Write-Host "QuantEdge AI started"
Write-Host "Frontend: http://127.0.0.1:5173"
Write-Host "Local API: http://127.0.0.1:8000"
Write-Host "Processes: API=$($api.Id), Web=$($web.Id)"
