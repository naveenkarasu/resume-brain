#!/usr/bin/env bash
# Setup script: copy PyInstaller output into Tauri sidecar location.
# Run from the desktop/ directory after building the backend exe.
#
# Usage:
#   cd resume-brain/desktop
#   bash setup-sidecar.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIST="$SCRIPT_DIR/../backend/dist/resume-brain"
SIDECAR_DIR="$SCRIPT_DIR/src-tauri/binaries"

if [ ! -d "$BACKEND_DIST" ]; then
    echo "[ERROR] Backend dist not found at: $BACKEND_DIST"
    echo "        Run 'python build_exe.py' in the backend/ directory first."
    exit 1
fi

echo "[SETUP] Copying backend dist to sidecar location..."
echo "        From: $BACKEND_DIST"
echo "        To:   $SIDECAR_DIR"

# Clean existing sidecar
rm -rf "$SIDECAR_DIR"
mkdir -p "$SIDECAR_DIR"

# Get the target triple for the current platform
TRIPLE=$(rustc -vV 2>/dev/null | grep '^host:' | sed 's/host: //')
if [ -z "$TRIPLE" ]; then
    # Fallback based on OS
    if [ "$(uname -s)" = "Darwin" ]; then
        if [ "$(uname -m)" = "arm64" ]; then
            TRIPLE="aarch64-apple-darwin"
        else
            TRIPLE="x86_64-apple-darwin"
        fi
    else
        TRIPLE="x86_64-unknown-linux-gnu"
    fi
    echo "[WARN] Could not detect Rust target triple, using default: $TRIPLE"
fi

# Copy the binary with the triple suffix (Tauri convention)
cp "$BACKEND_DIST/resume-brain" "$SIDECAR_DIR/resume-brain-$TRIPLE"
chmod +x "$SIDECAR_DIR/resume-brain-$TRIPLE"

# Copy the _internal directory (PyInstaller dependencies)
if [ -d "$BACKEND_DIST/_internal" ]; then
    cp -r "$BACKEND_DIST/_internal" "$SIDECAR_DIR/_internal"
fi

# Copy any other files/dirs from dist (exclude the binary and _internal we already copied)
for item in "$BACKEND_DIST"/*; do
    name="$(basename "$item")"
    if [ "$name" != "resume-brain" ] && [ "$name" != "_internal" ]; then
        cp -r "$item" "$SIDECAR_DIR/$name"
    fi
done

EXE_PATH="$SIDECAR_DIR/resume-brain-$TRIPLE"
if [ -f "$EXE_PATH" ]; then
    SIZE=$(du -h "$EXE_PATH" | cut -f1)
    echo ""
    echo "[OK] Sidecar setup complete!"
    echo "     Binary: $EXE_PATH ($SIZE)"
    echo "     Target triple: $TRIPLE"
else
    echo "[ERROR] Sidecar binary not found after copy!"
    exit 1
fi
