#!/bin/sh
# Ash Runtime Bootstrap Script
#
# Downloads and starts ash-runtime inside any container.
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/dreamyang-liu/Ash/main/runtime/bootstrap.sh | sh
#
# Or with custom setup before starting:
#   curl -fsSL .../bootstrap.sh | ASH_SETUP="cd /workspace && pip install -e ." sh
#
# Environment variables:
#   ASH_VERSION:   Release version (default: latest)
#   ASH_PORT:      Port to listen on (default: 3000)
#   ASH_SETUP:     Command to run before starting ash-runtime
#   ASH_ARCH:      Architecture (default: auto-detected, amd64 or arm64)
#   ASH_BIN_URL:   Direct URL to ash-runtime binary (skips GitHub release)

set -e

ASH_VERSION="${ASH_VERSION:-latest}"
ASH_PORT="${ASH_PORT:-3000}"
ASH_ARCH="${ASH_ARCH:-$(uname -m | sed 's/x86_64/amd64/' | sed 's/aarch64/arm64/')}"
ASH_OS="${ASH_OS:-$(uname -s | tr '[:upper:]' '[:lower:]')}"
ASH_BIN="/usr/local/bin/ash-runtime"

# Download binary
if [ -n "$ASH_BIN_URL" ]; then
    echo "[ash] downloading from $ASH_BIN_URL"
    curl -fsSL "$ASH_BIN_URL" -o "$ASH_BIN"
elif [ "$ASH_VERSION" = "latest" ]; then
    URL="https://github.com/dreamyang-liu/Ash/releases/latest/download/ash-runtime-${ASH_OS}-${ASH_ARCH}"
    echo "[ash] downloading latest from $URL"
    curl -fsSL "$URL" -o "$ASH_BIN" || {
        echo "[ash] release download failed, trying build from source..."
        if command -v go >/dev/null 2>&1; then
            git clone --depth 1 https://github.com/dreamyang-liu/Ash.git /tmp/ash-src
            cd /tmp/ash-src/runtime && go build -o "$ASH_BIN" .
        else
            echo "[ash] ERROR: cannot download or build ash-runtime"
            exit 1
        fi
    }
else
    URL="https://github.com/dreamyang-liu/Ash/releases/download/v${ASH_VERSION}/ash-runtime-${ASH_OS}-${ASH_ARCH}"
    echo "[ash] downloading v${ASH_VERSION} from $URL"
    curl -fsSL "$URL" -o "$ASH_BIN"
fi

chmod +x "$ASH_BIN"
echo "[ash] installed $(${ASH_BIN} --help 2>&1 | head -1 || echo 'ash-runtime')"

# Run user setup command if provided
if [ -n "$ASH_SETUP" ]; then
    echo "[ash] running setup: $ASH_SETUP"
    eval "$ASH_SETUP"
fi

# Start runtime
echo "[ash] starting on port $ASH_PORT"
exec "$ASH_BIN" --port "$ASH_PORT"
