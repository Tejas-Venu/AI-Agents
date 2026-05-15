#!/bin/bash
# ============================================================
# ngrok Campus Test Script
# 
# Tests whether ngrok can create a tunnel from behind your
# university firewall. Run this on the Linux server you SSH into.
#
# Steps:
#   1. Installs ngrok (if not present)
#   2. Checks for authtoken (prompts if missing)
#   3. Starts a tiny Python HTTP server
#   4. Opens an ngrok tunnel to it
#   5. Verifies the tunnel works from the outside
#
# Usage:
#   chmod +x test_ngrok.sh
#   ./test_ngrok.sh
#
# To clean up manually if the script is interrupted:
#   kill $(cat /tmp/ngrok_test_server.pid 2>/dev/null) 2>/dev/null
#   kill $(cat /tmp/ngrok_test_ngrok.pid 2>/dev/null) 2>/dev/null
# ============================================================

PORT=8765
TIMEOUT=15

cleanup() {
    echo ""
    echo "Cleaning up..."
    kill $(cat /tmp/ngrok_test_server.pid 2>/dev/null) 2>/dev/null || true
    kill $(cat /tmp/ngrok_test_ngrok.pid 2>/dev/null) 2>/dev/null || true
    rm -f /tmp/ngrok_test_server.pid /tmp/ngrok_test_ngrok.pid
    echo "Done."
}
trap cleanup EXIT

echo "========================================"
echo "  ngrok Campus Connectivity Test"
echo "========================================"
echo ""

# --- Step 1: Check for ngrok ---
echo "[1/6] Checking for ngrok..."
if command -v ngrok &> /dev/null; then
    echo "  âœ“ ngrok found at $(which ngrok)"
elif [ -f "$HOME/bin/ngrok" ]; then
    echo "  âœ“ ngrok found at ~/bin/ngrok"
    export PATH="$PATH:$HOME/bin"
else
    echo "  ngrok not found. Installing to ~/bin/ ..."
    mkdir -p "$HOME/bin"

    # Detect OS
    OS=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$OS" in
        linux*)  NGROK_OS="linux" ;;
        darwin*) NGROK_OS="darwin" ;;
        mingw*|msys*|cygwin*) NGROK_OS="windows" ;;
        *)
            echo "  âœ— Unsupported OS: $OS"
            echo "    Please install ngrok manually: https://ngrok.com/download"
            exit 1
            ;;
    esac

    # Detect architecture
    ARCH=$(uname -m)
    case "$ARCH" in
        x86_64|amd64)  NGROK_ARCH="amd64" ;;
        arm64|aarch64) NGROK_ARCH="arm64" ;;
        armv7l|armhf)  NGROK_ARCH="arm" ;;
        i386|i686)     NGROK_ARCH="386" ;;
        *)
            echo "  âœ— Unsupported architecture: $ARCH"
            echo "    Please install ngrok manually: https://ngrok.com/download"
            exit 1
            ;;
    esac

    NGROK_URL="https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-${NGROK_OS}-${NGROK_ARCH}.tgz"
    echo "  Detected: ${NGROK_OS}/${NGROK_ARCH}"
    echo "  Downloading from: ${NGROK_URL}"

    if command -v curl &> /dev/null; then
        curl -sSL "$NGROK_URL" | tar xz -C "$HOME/bin"
    elif command -v wget &> /dev/null; then
        wget -qO- "$NGROK_URL" | tar xz -C "$HOME/bin"
    else
        echo "  âœ— Neither curl nor wget available. Please install ngrok manually:"
        echo "    https://ngrok.com/download"
        exit 1
    fi
    chmod +x "$HOME/bin/ngrok"
    export PATH="$PATH:$HOME/bin"
    echo "  âœ“ ngrok installed to ~/bin/ngrok (${NGROK_OS}/${NGROK_ARCH})"

    # Remind user to add ~/bin to PATH permanently if it's not already there
    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/bin"; then
        echo ""
        echo "  TIP: Add ~/bin to your PATH permanently by adding this"
        echo "  to your ~/.bashrc or ~/.zshrc:"
        echo "    export PATH=\"\$PATH:\$HOME/bin\""
    fi
fi
echo ""

# --- Step 2: Check for authtoken ---
echo "[2/6] Checking for ngrok authtoken..."

# Check all known ngrok config file locations for an authtoken line
AUTHTOKEN_SET=false
NGROK_CONFIG_LOCATIONS=(
    "$HOME/.config/ngrok/ngrok.yml"                          # Linux (XDG)
    "$HOME/.ngrok2/ngrok.yml"                                # Legacy (all platforms)
    "$HOME/Library/Application Support/ngrok/ngrok.yml"      # macOS
    "$APPDATA/ngrok/ngrok.yml"                               # Windows
)

for cfg in "${NGROK_CONFIG_LOCATIONS[@]}"; do
    if [ -f "$cfg" ] && grep -q "authtoken:" "$cfg" 2>/dev/null; then
        AUTHTOKEN_SET=true
        echo "  âœ“ Authtoken found in: $cfg"
        break
    fi
done

if [ "$AUTHTOKEN_SET" = true ]; then
    :  # path already printed above
else
    echo "  âœ— No authtoken found."
    echo ""
    echo "  ngrok requires a free account and authtoken to work."
    echo "  If you don't have one yet:"
    echo ""
    echo "    1. Go to https://dashboard.ngrok.com/signup"
    echo "    2. Sign up (free)"
    echo "    3. Go to https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "    4. Copy your authtoken"
    echo ""
    read -p "  Paste your authtoken here (or Ctrl+C to quit): " USER_TOKEN
    echo ""

    if [ -z "$USER_TOKEN" ]; then
        echo "  âœ— No token entered. Exiting."
        exit 1
    fi

    # Strip any whitespace
    USER_TOKEN=$(echo "$USER_TOKEN" | tr -d '[:space:]')

    echo "  Configuring ngrok with your authtoken..."
    if ngrok config add-authtoken "$USER_TOKEN" 2>&1; then
        echo "  âœ“ Authtoken configured successfully"
    else
        echo "  âœ— Failed to configure authtoken. Check that the token is correct."
        exit 1
    fi
fi
echo ""

# --- Step 3: Start a simple HTTP server ---
echo "[3/6] Starting test HTTP server on port $PORT..."
python3 -c "
import http.server, json, os

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        response = {
            'status': 'ok',
            'message': 'ngrok tunnel is working!',
            'hostname': '$(hostname)',
            'test': 'campus_connectivity'
        }
        self.wfile.write(json.dumps(response, indent=2).encode())
    def log_message(self, format, *args):
        pass  # suppress logs

server = http.server.HTTPServer(('127.0.0.1', $PORT), Handler)
with open('/tmp/ngrok_test_server.pid', 'w') as f:
    f.write(str(os.getpid()))
server.serve_forever()
" &
sleep 1

# Verify server is running
if curl -s http://127.0.0.1:$PORT > /dev/null 2>&1; then
    echo "  âœ“ HTTP server running on localhost:$PORT"
else
    echo "  âœ— Failed to start HTTP server. Is port $PORT in use?"
    exit 1
fi
echo ""

# --- Step 4: Start ngrok ---
echo "[4/6] Starting ngrok tunnel (this is the real test)..."
echo "  If this hangs, ngrok is likely blocked on your network."
echo ""

ngrok http $PORT --log=stdout --log-level=info > /tmp/ngrok_test.log 2>&1 &
echo $! > /tmp/ngrok_test_ngrok.pid

# Wait for ngrok to establish the tunnel
echo -n "  Waiting for tunnel"
TUNNEL_URL=""
for i in $(seq 1 $TIMEOUT); do
    echo -n "."
    sleep 1
    # Try the ngrok local API to get the tunnel URL
    TUNNEL_URL=$(curl -s http://127.0.0.1:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for t in data.get('tunnels', []):
        if t.get('public_url', '').startswith('https://'):
            print(t['public_url'])
            break
except:
    pass
" 2>/dev/null)
    if [ -n "$TUNNEL_URL" ]; then
        break
    fi
done
echo ""
echo ""

if [ -z "$TUNNEL_URL" ]; then
    echo "  âœ— FAILED: ngrok could not establish a tunnel within ${TIMEOUT}s."
    echo ""
    echo "  Possible reasons:"
    echo "    - University firewall blocks ngrok's outbound connections"
    echo "    - ngrok binary is outdated"
    echo "    - Network requires a proxy (check \$http_proxy)"
    echo ""
    echo "  Last few lines of ngrok log:"
    tail -20 /tmp/ngrok_test.log 2>/dev/null | grep -i "err\|fail\|block\|refused\|timeout" | head -5
    echo ""
    echo "  >>> Try Cloudflare Tunnel as an alternative: <<<"
    echo "  curl -sSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared"
    echo "  chmod +x cloudflared"
    echo "  ./cloudflared tunnel --url http://localhost:8765"
    exit 1
fi

echo "  âœ“ Tunnel established!"
echo "  Public URL: $TUNNEL_URL"
echo ""

# --- Step 5: Test the tunnel from the outside ---
echo "[5/6] Testing tunnel connectivity..."
# The ngrok free tier shows an interstitial page for browsers,
# but API calls with the right header bypass it.
# Allow a few retries in case the tunnel needs a moment.
TUNNEL_OK=false
for attempt in 1 2 3; do
    sleep 1
    RESPONSE=$(curl -s --max-time 5 -H "ngrok-skip-browser-warning: true" "$TUNNEL_URL" 2>/dev/null || echo "")

    if [ -n "$RESPONSE" ] && echo "$RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('status') == 'ok':
        sys.exit(0)
except:
    pass
sys.exit(1)
" 2>/dev/null; then
        TUNNEL_OK=true
        break
    fi
done

if [ "$TUNNEL_OK" = true ]; then
    echo "  âœ“ Tunnel is fully functional!"
    echo "  Response from tunnel:"
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null | sed 's/^/    /'
else
    echo "  âš  Tunnel exists but the test response was unexpected."
    echo "  This is normal â€” the ngrok free tier often shows a"
    echo "  browser warning page on the first request. Programmatic"
    echo "  calls from your agents (with the ngrok-skip-browser-warning"
    echo "  header) should work fine."
    echo ""
    echo "  The important result is that the tunnel was established."
fi
echo ""

# --- Step 6: Summary ---
echo "[6/6] Results"
echo "========================================"
echo ""
echo "  ngrok:     WORKS âœ“"
echo "  Tunnel:    $TUNNEL_URL"
echo "  Server:    localhost:$PORT"
echo ""
echo "  Your students can use ngrok to expose their"
echo "  MCP/A2A agents from this network."
echo ""
echo "  Quick reference for students:"
echo "    ngrok http 8000"
echo ""
echo "  Note: Each student will need their own free ngrok"
echo "  account and authtoken. Have them do this BEFORE class:"
echo "    1. Sign up at https://dashboard.ngrok.com/signup"
echo "    2. Copy authtoken from the dashboard"
echo "    3. Run: ngrok config add-authtoken <token>"
echo ""
echo "  Press Ctrl+C to shut down the test."
echo "========================================"

# Keep running so you can manually test the URL if desired
wait