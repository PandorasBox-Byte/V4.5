#!/usr/bin/env bash
# Simple packager: creates a macOS .app bundle that contains a copy of the
# project and a virtualenv inside the bundle. This is a convenience helper
# for local distribution; it does not create a signed or notarized app.

set -euo pipefail

APP_NAME="EvoAI"
OUT_DIR="dist"
APP_DIR="$OUT_DIR/${APP_NAME}.app"

if [ -d "$APP_DIR" ]; then
  echo "Removing existing $APP_DIR"
  rm -rf "$APP_DIR"
fi

mkdir -p "$APP_DIR/Contents/MacOS"
mkdir -p "$APP_DIR/Contents/Resources/app"

echo "Copying project files into app bundle..."
# Copy project, excluding large virtualenvs and dist
rsync -a --exclude='.venv*' --exclude='dist' --exclude='*.pyc' --exclude='__pycache__' ./ "$APP_DIR/Contents/Resources/app/"

echo "Creating virtualenv inside bundle..."
python3 -m venv "$APP_DIR/Contents/Resources/venv"
source "$APP_DIR/Contents/Resources/venv/bin/activate"
pip install --upgrade pip setuptools wheel
if [ -f "requirements.txt" ]; then
  pip install -r "$APP_DIR/Contents/Resources/app/requirements.txt"
fi

echo "Writing launcher stub..."
cat > "$APP_DIR/Contents/MacOS/$APP_NAME" <<'EOF'
#!/usr/bin/env bash
DIR="$(cd "$(dirname "$0")/../Resources/app" && pwd)"
VENV="$(cd "$(dirname "$0")/../Resources/venv" && pwd)"
export PATH="$VENV/bin:$PATH"
cd "$DIR"
PYTHONPATH="$DIR" "$VENV/bin/python" core/launcher.py
EOF
chmod +x "$APP_DIR/Contents/MacOS/$APP_NAME"

echo "Writing Info.plist..."
cat > "$APP_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>$APP_NAME</string>
  <key>CFBundleExecutable</key>
  <string>$APP_NAME</string>
  <key>CFBundleIdentifier</key>
  <string>com.example.$APP_NAME</string>
  <key>CFBundleVersion</key>
  <string>0.1</string>
  <key>LSBackgroundOnly</key>
  <false/>
</dict>
</plist>
EOF

echo "Created $APP_DIR — you can distribute this folder as an app bundle.
To run: open $APP_DIR or execute $APP_DIR/Contents/MacOS/$APP_NAME"
