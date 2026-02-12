# Setup script: copy PyInstaller output into Tauri sidecar location.
# Run from the desktop/ directory after building the backend exe.
#
# Usage:
#   cd resume-brain/desktop
#   powershell -ExecutionPolicy Bypass -File setup-sidecar.ps1

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendDist = Join-Path $ScriptDir "..\backend\dist\resume-brain"
$SidecarDir = Join-Path $ScriptDir "src-tauri\binaries"

if (-not (Test-Path $BackendDist)) {
    Write-Host "[ERROR] Backend dist not found at: $BackendDist" -ForegroundColor Red
    Write-Host "        Run 'python build_exe.py' in the backend/ directory first." -ForegroundColor Yellow
    exit 1
}

Write-Host "[SETUP] Copying backend dist to sidecar location..."
Write-Host "        From: $BackendDist"
Write-Host "        To:   $SidecarDir"

# Clean existing sidecar
if (Test-Path $SidecarDir) {
    Remove-Item -Recurse -Force $SidecarDir
}
New-Item -ItemType Directory -Force -Path $SidecarDir | Out-Null

# Copy entire dist directory
# Tauri expects the sidecar exe at binaries/resume-brain-x86_64-pc-windows-msvc.exe
# The _internal directory with all dependencies must be alongside it

# Get the target triple for the current platform
$Triple = rustc -vV 2>$null | Select-String "host:" | ForEach-Object { $_.Line -replace "host:\s*", "" }
if (-not $Triple) {
    $Triple = "x86_64-pc-windows-msvc"
    Write-Host "[WARN] Could not detect Rust target triple, using default: $Triple" -ForegroundColor Yellow
}

# Copy the exe with the triple suffix (Tauri convention)
Copy-Item (Join-Path $BackendDist "resume-brain.exe") (Join-Path $SidecarDir "resume-brain-$Triple.exe")

# Copy the _internal directory (PyInstaller dependencies)
$InternalDir = Join-Path $BackendDist "_internal"
if (Test-Path $InternalDir) {
    Copy-Item -Recurse $InternalDir (Join-Path $SidecarDir "_internal")
}

# Copy any other files/dirs from dist (exclude the exe we already copied)
Get-ChildItem $BackendDist -Exclude "resume-brain.exe", "_internal" | ForEach-Object {
    Copy-Item -Recurse $_.FullName (Join-Path $SidecarDir $_.Name)
}

$ExePath = Join-Path $SidecarDir "resume-brain-$Triple.exe"
if (Test-Path $ExePath) {
    $Size = (Get-Item $ExePath).Length / 1MB
    Write-Host ""
    Write-Host "[OK] Sidecar setup complete!" -ForegroundColor Green
    Write-Host "     Exe: $ExePath ($([math]::Round($Size, 1)) MB)"
    Write-Host "     Target triple: $Triple"
} else {
    Write-Host "[ERROR] Sidecar exe not found after copy!" -ForegroundColor Red
    exit 1
}
