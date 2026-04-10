<#
.SYNOPSIS
    Start, stop, or check the DAFoam Chroma RAG SSE MCP server (port 29310).

.EXAMPLE
    .\scripts\Start-ChromaServer.ps1           # Start the server
    .\scripts\Start-ChromaServer.ps1 -Stop     # Stop the server
    .\scripts\Start-ChromaServer.ps1 -Status   # Check if running
    .\scripts\Start-ChromaServer.ps1 -Port 29311  # Custom port
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Status,
    [int]$Port = 0
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PidFile     = Join-Path $ProjectRoot "data\chroma_server.pid"
$LogFile     = Join-Path $ProjectRoot "data\chroma_server.log"
$ErrFile     = Join-Path $ProjectRoot "data\chroma_server_err.log"
$ServerScript = Join-Path $ProjectRoot "src\mcp\chroma_sse_server.py"
$VenvPython  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

# Resolve python executable
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

# Read stored PID
function Get-StoredPid {
    if (Test-Path $PidFile) {
        $raw = (Get-Content $PidFile -Raw).Trim()
        if ($raw -match '^\d+$') { return [int]$raw }
    }
    return $null
}

function Is-Running([int]$ProcessId) {
    try { $proc = Get-Process -Id $ProcessId -ErrorAction Stop; return $true }
    catch { return $false }
}

# ── Status ────────────────────────────────────────────────────────────────────
if ($Status) {
    $storedPid = Get-StoredPid
    if ($null -ne $storedPid -and (Is-Running $storedPid)) {
        Write-Host "dafoam-rag SSE server is RUNNING (PID $storedPid)" -ForegroundColor Green
        $effectivePort = if ($Port -gt 0) { $Port } else { 29310 }
        Write-Host "  SSE URL: http://127.0.0.1:$effectivePort/sse"
    } else {
        Write-Host "dafoam-rag SSE server is NOT running." -ForegroundColor Yellow
    }
    exit
}

# ── Stop ──────────────────────────────────────────────────────────────────────
if ($Stop) {
    $storedPid = Get-StoredPid
    if ($null -ne $storedPid -and (Is-Running $storedPid)) {
        Stop-Process -Id $storedPid -Force
        Remove-Item -Path $PidFile -ErrorAction SilentlyContinue
        Write-Host "dafoam-rag SSE server stopped (PID $storedPid)." -ForegroundColor Yellow
    } else {
        Write-Host "No running server found." -ForegroundColor Gray
    }
    exit
}

# ── Start ─────────────────────────────────────────────────────────────────────
$storedPid = Get-StoredPid
if ($null -ne $storedPid -and (Is-Running $storedPid)) {
    Write-Host "Server already running (PID $storedPid). Use -Status to verify or -Stop to stop." -ForegroundColor Cyan
    exit
}

# Ensure data/ directory exists
New-Item -ItemType Directory -Force -Path (Split-Path $PidFile) | Out-Null

$portArgs = @()
if ($Port -gt 0) { $portArgs = @("--port", $Port) }

Push-Location $ProjectRoot
$proc = Start-Process -FilePath $Python `
    -ArgumentList (@($ServerScript) + $portArgs) `
    -RedirectStandardOutput $LogFile `
    -RedirectStandardError  $ErrFile `
    -PassThru -WindowStyle Hidden

$proc.Id | Set-Content -Path $PidFile
$effectivePort = if ($Port -gt 0) { $Port } else { 29310 }

Write-Host "dafoam-rag SSE server started (PID $($proc.Id))." -ForegroundColor Green
Write-Host "  SSE URL:  http://127.0.0.1:$effectivePort/sse"
Write-Host "  Logs:     $LogFile"
Write-Host "  Errors:   $ErrFile"
Write-Host ""
Write-Host "Add to ~/.copilot/mcp-config.json:"
Write-Host "  `"dafoam-rag`": { `"type`": `"sse`", `"url`": `"http://127.0.0.1:$effectivePort/sse`" }"
Pop-Location
