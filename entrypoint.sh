#!/bin/bash

echo "======================================="
echo "                      _          _                "
echo "                     | |        (_)               "
echo "  _ __ ___   ___  ___| |____   ___  _____      __ "
echo " | '_ \` _ \ / _ \/ __| '_ \ \ / / |/ _ \ \ /\ / / "
echo " | | | | | |  __/\__ \ | | \ V /| |  __/\ V  V /  "
echo " |_| |_| |_|\___||___/_| |_|\_/ |_|\___| \_/\_/   "
echo "                                                 "
echo "        Meshtastic Meshview 🚀                  "
echo "======================================="
echo ""

# Verify config.ini is present
if [ ! -f "/app/config.ini" ]; then
    echo "[❌ ERROR] Missing config.ini. Please mount it into the container."
    exit 1
fi

# Start Database in Background
echo "[✅ INFO] Starting Meshview Database..."
python startdb.py --config config.ini &

# Brief wait for DB initialization
sleep 5

# Start Web Server
echo "[🌐 INFO] Launching Meshview Webserver..."
python main.py --config config.ini
