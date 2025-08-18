#!/usr/bin/env bash
set -euo pipefail

# === Script Params ===
APP_NAME="AutoSignatureDetector"
ENTRY="main.py"
BUNDLE_DIR="${APP_NAME}"
VENV_DIR=".venv"

# Activate venv
source "$VENV_DIR/bin/activate"

# Install requirements.txt
if [ -f "requirements.txt" ]; then
  pip install -r requirements.txt

# Clean old builds
rm -rf "$BUNDLE_DIR" build dist "${APP_NAME}.spec"

# Build (using PyInstaller create exec file)
pyinstaller --onefile --name "$APP_NAME" "$ENTRY"

# Prepare bundle folder
mkdir -p "$BUNDLE_DIR"

# Copy executable
if [ -f "dist/${APP_NAME}" ]; then
  cp "dist/${APP_NAME}" "$BUNDLE_DIR/"
elif [ -f "dist/${APP_NAME}.exe" ]; then
  cp "dist/${APP_NAME}.exe" "$BUNDLE_DIR/"
else
  echo "ERROR: built executable not found in dist/"
  exit 2
fi

# Copy config.toml
if [ -f "config.toml" ]; then
  cp "config.toml" "$BUNDLE_DIR/config.toml"
else
  echo "Warning: config.toml not found in project root"
fi

# Copy templates_pdf folder
if [ -d "templates_pdf" ]; then
  cp -r "templates_pdf" "$BUNDLE_DIR/templates_pdf"
else
  echo "Warning: templates_pdf folder not found in project root"
fi

# Create empty to_process folder
mkdir -p "$BUNDLE_DIR/to_process"

# Make executable runnable
chmod +x "$BUNDLE_DIR/${APP_NAME}" || true

echo "#############################################"
echo "Executable created check folder: $BUNDLE_DIR"
echo "#############################################"
