#!/bin/bash
# ROBOVANGUARD WRO 2026 Autostart Installation Script for Raspberry Pi 5

echo "========================================================="
echo "   ROBOVANGUARD WRO 2026 - Pi 5 Autostart Installer"
echo "========================================================="

# Automatically detect current script directory path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="$(which python3)"
SYSTEM_USER="$USER"

if [ -z "$SYSTEM_USER" ]; then
    SYSTEM_USER="pi"
fi

echo "[INFO] Repository Directory: $SCRIPT_DIR"
echo "[INFO] Python Executable:    $PYTHON_EXEC"
echo "[INFO] System User:          $SYSTEM_USER"

SERVICE_FILE="/etc/systemd/system/wro_autostart.service"

# Create systemd service with dynamic path detection
cat << EOF | sudo tee $SERVICE_FILE > /dev/null
[Unit]
Description=ROBOVANGUARD WRO 2026 Competition Autostart Service
After=multi-user.target serial-getty@ttyUSB0.service

[Service]
Type=simple
User=$SYSTEM_USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=$PYTHON_EXEC $SCRIPT_DIR/competition_launcher.py --pin 17
Restart=on-failure
RestartSec=3s
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd daemon & enable service
sudo systemctl daemon-reload
sudo systemctl enable wro_autostart.service

echo ""
echo "[SUCCESS] WRO Autostart Service successfully installed & enabled!"
echo "  -> Systemd will now automatically launch competition_launcher.py on boot."
echo "  -> To test immediately: sudo systemctl start wro_autostart"
echo "  -> To check logs/status: sudo systemctl status wro_autostart"
echo "========================================================="
