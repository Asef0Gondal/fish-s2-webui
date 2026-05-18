#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Fish S2 Pro — Install Script (run once)
# ─────────────────────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

echo ""
echo "🐟  Fish S2 Pro — WebUI Installer"
echo "─────────────────────────────────────────────"
echo ""

# Check Apple Silicon
ARCH=$(uname -m)
if [ "$ARCH" != "arm64" ]; then
  echo "❌  This app requires Apple Silicon (M-series Mac)."
  echo "    Detected: $ARCH"
  read -p "Press Enter to close..."
  exit 1
fi
echo "✅  Apple Silicon detected ($ARCH)"

# Find Python 3.11+
PYTHON=""
for PY in /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11; do
  if [ -f "$PY" ]; then
    PYTHON="$PY"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  echo ""
  echo "❌  Python 3.11+ not found via Homebrew."
  echo "    Run: brew install python@3.13"
  read -p "Press Enter to close..."
  exit 1
fi

echo "✅  Python found: $PYTHON ($($PYTHON --version))"
echo ""

# Create virtual environment
echo "📦  Creating virtual environment..."
"$PYTHON" -m venv "$DIR/venv"
source "$DIR/venv/bin/activate"

echo "📦  Upgrading pip..."
pip install --upgrade pip -q

echo ""
echo "📦  Installing dependencies..."
echo "    (fastapi, uvicorn, mlx-speech, soundfile, python-multipart)"
echo ""

pip install \
  mlx-speech \
  fastapi \
  "uvicorn[standard]" \
  python-multipart \
  soundfile \
  numpy

if [ $? -ne 0 ]; then
  echo ""
  echo "❌  Installation failed. Check the error above."
  read -p "Press Enter to close..."
  exit 1
fi

echo ""
echo "─────────────────────────────────────────────"
echo "📥  Pre-downloading Fish S2 Pro model..."
echo "    (~7 GB — this will take a few minutes on first run)"
echo "    The model is saved to ~/.cache/mlx-speech/"
echo ""

python -c "
import mlx_speech
print('Downloading model weights...')
try:
    m = mlx_speech.tts.load('fish-s2-pro')
    print('✅  Model downloaded and verified.')
except Exception as e:
    print(f'⚠  Model pre-download failed: {e}')
    print('   It will download automatically on first launch.')
"

echo ""
echo "🔐  Making launch.command executable..."
chmod +x "$DIR/launch.command"

echo ""
echo "─────────────────────────────────────────────"
echo "✅  Installation complete!"
echo ""
echo "👉  To start the app:"
echo "    Double-click  launch.command"
echo ""
echo "📁  Voice library location:"
echo "    $DIR/voices/"
echo ""
echo "📁  Generated audio location:"
echo "    $DIR/outputs/"
echo ""
read -p "Press Enter to close..."
