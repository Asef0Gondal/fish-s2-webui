#!/bin/bash
# ─────────────────────────────────────────────────────────────
# Fish S2 Pro — Local WebUI Launcher
# Double-click this file to start. Close this window to stop.
# ─────────────────────────────────────────────────────────────

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Check venv exists
if [ ! -f "$DIR/venv/bin/python" ]; then
  echo "❌  Virtual environment not found."
  echo "    Please run install.sh first."
  echo ""
  read -p "Press Enter to close..."
  exit 1
fi

source "$DIR/venv/bin/activate"

# Kill any leftover process on port 8765
EXISTING=$(lsof -ti:8765 2>/dev/null)
if [ -n "$EXISTING" ]; then
  echo "⚠  Killing existing process on port 8765..."
  echo "$EXISTING" | xargs kill -9 2>/dev/null
  sleep 1
fi

echo ""
echo "🐟  Fish S2 Pro — Local WebUI"
echo "─────────────────────────────────"
echo "📍  Starting server at http://localhost:8765"
echo "🧠  Loading Fish S2 Pro model (this takes ~10–20 sec)..."
echo ""
echo "💡  The browser will open automatically once the server is ready."
echo "🛑  Close this window to stop the server and unload the model."
echo "─────────────────────────────────"
echo ""

# Start server using venv Python explicitly (source activate is unreliable in .command files)
PYTHON="$DIR/venv/bin/python"
"$PYTHON" "$DIR/app.py" &
SERVER_PID=$!

# Wait for server to respond (poll /api/status)
echo -n "    Waiting for server"
for i in $(seq 1 40); do
  sleep 1
  echo -n "."
  STATUS=$(curl -s http://localhost:8765/api/status 2>/dev/null)
  if echo "$STATUS" | grep -q '"status"'; then
    break
  fi
done
echo ""
echo ""

# Open browser
open http://localhost:8765

echo "✅  Server is running at http://localhost:8765"
echo ""

# Keep script alive — when window closes, kill server
trap "echo ''; echo '🛑 Shutting down...'; kill $SERVER_PID 2>/dev/null; exit 0" INT TERM EXIT

wait $SERVER_PID
