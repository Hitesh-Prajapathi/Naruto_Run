# One command to play -- Feature Brief 05 §3.2.
#
# "Ship a clean startup path: one command that launches both the Python
#  service and the game, plus a documented manual path. If starting the game
#  requires remembering two terminal commands in the right order, it will get
#  done wrong."
#
# Starts the recognition service, waits until it is actually answering, then
# starts the Vite dev server and opens the game.
#
#   .\cv_model\Nishit_Frontend\start_game.ps1
#
# Manual path (two terminals), if you want the logs separated:
#   1) .\cv_model\Nishit_Frontend\venv\Scripts\python.exe cv_model\Nishit_Frontend\serve_transport.py
#   2) cd cv_model\Nishit_Frontend\frontend ; npm run dev
#      then open http://localhost:5173/scene.html
#
# Keyboard-only, no camera and no service needed:
#   http://localhost:5173/scene.html?cv=0

[CmdletBinding()]
param(
    # Skip the recognition service; play with the keyboard.
    [switch]$KeyboardOnly,
    [int]$ServicePort = 8765,
    [int]$WebPort = 5173
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $here "venv\Scripts\python.exe"
$server = Join-Path $here "serve_transport.py"
$frontend = Join-Path $here "frontend"

function Test-PortOpen([int]$Port) {
    try {
        $client = New-Object Net.Sockets.TcpClient
        $client.Connect("127.0.0.1", $Port)
        $client.Close()
        return $true
    } catch {
        return $false
    }
}

$jobs = @()

if (-not $KeyboardOnly) {
    if (-not (Test-Path $python)) {
        Write-Error "Virtual environment missing at $python. See the Setup section of README.md."
    }
    if (Test-PortOpen $ServicePort) {
        Write-Host "Recognition service already running on port $ServicePort." -ForegroundColor Yellow
    } else {
        Write-Host "Starting recognition service (loads MediaPipe + ONNX, takes a few seconds)..." -ForegroundColor Cyan
        # --body-only: the game reads body movement only, and hand inference is
        # more than half the per-frame cost. Skipping it roughly doubles the
        # frame rate, which is what keeps samples from arriving stale.
        $jobs += Start-Process -FilePath $python -ArgumentList @($server, "--body-only") -PassThru -WorkingDirectory $here

        # The models take a while to load; do not open the game before the
        # socket answers, or the player meets "service not running" on a
        # service that is merely still starting.
        $deadline = (Get-Date).AddSeconds(90)
        while (-not (Test-PortOpen $ServicePort)) {
            if ((Get-Date) -gt $deadline) {
                Write-Error "Recognition service did not start within 90s. Run it manually to see its output."
            }
            Start-Sleep -Milliseconds 500
        }
        Write-Host "Recognition service is up." -ForegroundColor Green
    }
}

if (Test-PortOpen $WebPort) {
    Write-Host "Dev server already running on port $WebPort." -ForegroundColor Yellow
} else {
    Write-Host "Starting the game..." -ForegroundColor Cyan
    # npm is npm.cmd on Windows; Start-Process will not resolve a bare "npm".
    $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
    if (-not $npm) { $npm = (Get-Command npm -ErrorAction SilentlyContinue).Source }
    if (-not $npm) { Write-Error "npm not found on PATH. Install Node.js, or run 'npm run dev' manually in $frontend." }
    $jobs += Start-Process -FilePath $npm -ArgumentList "run", "dev" -PassThru -WorkingDirectory $frontend

    $deadline = (Get-Date).AddSeconds(60)
    while (-not (Test-PortOpen $WebPort)) {
        if ((Get-Date) -gt $deadline) {
            Write-Error "Dev server did not start within 60s."
        }
        Start-Sleep -Milliseconds 300
    }
}

# ENABLE_CV_INPUT ships off, so camera mode has to opt in explicitly.
$suffix = if ($KeyboardOnly) { "?cv=0" } else { "?cv=1" }
$url = "http://localhost:$WebPort/scene.html$suffix"
Write-Host "Opening $url" -ForegroundColor Green
Start-Process $url

Write-Host ""
Write-Host "Controls: lean left/right to change lane, jump to jump." -ForegroundColor Gray
Write-Host "  R  recalibrate      C  toggle camera preview      `  debug panel" -ForegroundColor Gray
Write-Host "  Keyboard always works: A/D or arrows, Space to jump." -ForegroundColor Gray
Write-Host ""
Write-Host "Press Ctrl+C to stop, then close the spawned windows." -ForegroundColor Gray
