# MeshView Installation on Ubuntu 24 LTS

This guide will help you install MeshView on Ubuntu 24 LTS Desktop safely without crashing your system.

## Prerequisites

- Ubuntu 24 LTS Desktop
- Internet connection
- At least 2GB free disk space
- Regular user account (not root)

## Installation Steps

### 1. Clone the repository (if you haven't already)

```bash
cd ~
git clone https://github.com/logans-stuff/meshview.git
cd meshview
```

### 2. Run the installation script

```bash
./install-ubuntu.sh
```

This script will:
- Install system dependencies (Python, SQLite, Graphviz)
- Create a Python virtual environment
- Install Python packages
- Create a sample configuration file
- Create helper scripts

**Note:** The script will ask for your sudo password to install system packages. This is safe and expected.

### 3. Configure MeshView

Edit the configuration file:

```bash
nano ~/meshview/config.ini
```

**Important settings to configure:**

- `[mqtt]` section: Set your MQTT broker and topics
- `[site]` section: Set your site title, domain, etc.
- `[database]` section: Database location (default is fine)

Save and exit (Ctrl+X, then Y, then Enter)

### 4. Run MeshView

```bash
cd ~/meshview
./run-meshview.sh
```

MeshView should now be running! Open your browser and go to:
- http://localhost:8080

Press Ctrl+C to stop MeshView.

## Optional: Run as a System Service

If you want MeshView to start automatically on boot:

```bash
cd ~/meshview
./setup-service.sh
```

Then manage it with:

```bash
# Start the service
sudo systemctl start meshview

# Stop the service
sudo systemctl stop meshview

# Check status
sudo systemctl status meshview

# Enable on boot
sudo systemctl enable meshview

# View logs
sudo journalctl -u meshview -f
```

## Troubleshooting

### Port Already in Use

If port 8080 is already in use, edit `config.ini` and change:

```ini
[site]
port = 8081
```

### Permission Errors

Make sure you're not running as root:

```bash
whoami  # Should NOT show "root"
```

### Missing Dependencies

If installation fails, try updating your system first:

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Database Issues

If you get database errors, make sure the database directory exists and is writable:

```bash
mkdir -p ~/meshview/data
chmod 755 ~/meshview/data
```

### View Logs

When running as a service:

```bash
sudo journalctl -u meshview -f --no-pager
```

When running manually, logs will appear in the terminal.

## Updating MeshView

To update to the latest version:

```bash
cd ~/meshview
git pull
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

If running as a service, restart it:

```bash
sudo systemctl restart meshview
```

## Uninstalling

To completely remove MeshView:

```bash
# Stop and disable service (if installed)
sudo systemctl stop meshview
sudo systemctl disable meshview
sudo rm /etc/systemd/system/meshview.service
sudo systemctl daemon-reload

# Remove installation directory
rm -rf ~/meshview
```

## Safety Features

This installation script is designed to be safe:

- ✅ Does NOT require running as root
- ✅ Uses Python virtual environment (isolated from system Python)
- ✅ Only installs minimal system packages
- ✅ No aggressive compilation or memory-intensive operations
- ✅ Clear error messages if something goes wrong

## Getting Help

If you encounter issues:

1. Check the logs (see Troubleshooting section)
2. Check your config.ini settings
3. Make sure MQTT broker is accessible
4. Open an issue on GitHub with logs and error messages
