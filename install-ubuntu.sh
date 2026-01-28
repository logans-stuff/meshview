#!/bin/bash
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}MeshView Installation Script for Ubuntu 24 LTS${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if running on Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    echo -e "${YELLOW}Warning: This script is designed for Ubuntu. Continuing anyway...${NC}"
fi

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${RED}Error: Please do not run this script as root/sudo${NC}"
    echo "Run it as a regular user. Sudo will be used only when necessary."
    exit 1
fi

INSTALL_DIR="$HOME/meshview"
VENV_DIR="$INSTALL_DIR/venv"
CONFIG_FILE="$INSTALL_DIR/config.ini"

echo -e "${GREEN}Step 1/6: Updating system packages${NC}"
sudo apt-get update

echo -e "${GREEN}Step 2/6: Installing system dependencies${NC}"
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    sqlite3 \
    graphviz \
    git

echo -e "${GREEN}Step 3/6: Creating Python virtual environment${NC}"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Removing old one...${NC}"
    rm -rf "$VENV_DIR"
fi

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo -e "${GREEN}Step 4/6: Installing Python dependencies${NC}"
pip install --upgrade pip
pip install -r "$INSTALL_DIR/requirements.txt"

echo -e "${GREEN}Step 5/6: Setting up configuration${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}No config.ini found. Creating from sample...${NC}"
    cp "$INSTALL_DIR/sample.config.ini" "$CONFIG_FILE"
    echo -e "${YELLOW}Please edit $CONFIG_FILE before running MeshView${NC}"
else
    echo -e "${GREEN}Config file already exists at $CONFIG_FILE${NC}"
fi

echo -e "${GREEN}Step 6/6: Creating run script${NC}"
cat > "$INSTALL_DIR/run-meshview.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
python3 -m meshview
EOF
chmod +x "$INSTALL_DIR/run-meshview.sh"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installation Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Edit the configuration file:"
echo "   nano $CONFIG_FILE"
echo ""
echo "2. Configure your MQTT settings and other options"
echo ""
echo "3. Run MeshView:"
echo "   cd $INSTALL_DIR"
echo "   ./run-meshview.sh"
echo ""
echo -e "${YELLOW}Optional: Set up as a systemd service${NC}"
echo "Run: $INSTALL_DIR/setup-service.sh"
echo ""

# Create optional service setup script
cat > "$INSTALL_DIR/setup-service.sh" << 'SERVICEEOF'
#!/bin/bash
set -e

if [ "$EUID" -eq 0 ]; then
    echo "Error: Please do not run this script as root/sudo"
    exit 1
fi

INSTALL_DIR="$HOME/meshview"
SERVICE_FILE="/etc/systemd/system/meshview.service"

echo "Creating systemd service..."

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=MeshView - Meshtastic Network Viewer
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python3 -m meshview
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading systemd..."
sudo systemctl daemon-reload

echo ""
echo "Service created! You can now:"
echo "  Start:   sudo systemctl start meshview"
echo "  Stop:    sudo systemctl stop meshview"
echo "  Status:  sudo systemctl status meshview"
echo "  Enable:  sudo systemctl enable meshview  (start on boot)"
echo "  Logs:    sudo journalctl -u meshview -f"
SERVICEEOF
chmod +x "$INSTALL_DIR/setup-service.sh"

echo -e "${GREEN}Installation script created successfully!${NC}"
