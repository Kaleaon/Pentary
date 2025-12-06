#!/bin/bash
# Pentary Raspberry Pi Setup Script
# Installs and configures Pentary processor emulator on Raspberry Pi

set -e

echo "=========================================="
echo "Pentary Raspberry Pi Setup"
echo "=========================================="

# Check if running on Raspberry Pi
if [ ! -f /proc/device-tree/model ]; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for root
if [ "$EUID" -ne 0 ]; then 
    echo "Please run as root (use sudo)"
    exit 1
fi

echo ""
echo "Step 1: Updating system..."
apt update
apt upgrade -y

echo ""
echo "Step 2: Installing dependencies..."
apt install -y \
    python3-pip \
    python3-numpy \
    python3-scipy \
    python3-matplotlib \
    python3-dev \
    git \
    build-essential \
    cmake \
    cython3 \
    libatlas-base-dev \
    libopenblas-dev

echo ""
echo "Step 3: Installing Python packages..."
pip3 install --upgrade pip
pip3 install \
    torch \
    tensorflow-lite \
    pillow \
    opencv-python \
    pyserial \
    pyyaml \
    flask \
    paho-mqtt

echo ""
echo "Step 4: Installing Pentary software..."

# Create installation directory
mkdir -p /opt/pentary
cp -r tools/* /opt/pentary/

# Create symbolic links
ln -sf /opt/pentary/pentary_cli.py /usr/local/bin/pentary
ln -sf /opt/pentary/pentary_simulator.py /usr/local/bin/pentary-sim
chmod +x /usr/local/bin/pentary
chmod +x /usr/local/bin/pentary-sim

echo ""
echo "Step 5: Creating demo scripts..."

# Create pentary-demo script
cat > /usr/local/bin/pentary-demo << 'EOF'
#!/usr/bin/env python3
"""Pentary demo launcher"""
import sys
sys.path.insert(0, '/opt/pentary')
from pentary_cli import main
main(['demo'])
EOF
chmod +x /usr/local/bin/pentary-demo

# Create pentary-test script
cat > /usr/local/bin/pentary-test << 'EOF'
#!/usr/bin/env python3
"""Pentary test suite"""
import sys
sys.path.insert(0, '/opt/pentary')
from pentary_validation import run_tests
run_tests()
EOF
chmod +x /usr/local/bin/pentary-test

echo ""
echo "Step 6: Creating configuration..."
mkdir -p /etc/pentary
cat > /etc/pentary/config.yaml << 'EOF'
# Pentary Configuration
processor:
  num_registers: 32
  cache_size: 32768
  memory_size: 1048576

quantization:
  levels: 5
  range: [-2, 2]
  method: "symmetric"

performance:
  num_threads: 4
  use_simd: true
  use_cython: true

hardware:
  use_coral: false
  use_gpio: true
  gpio_pins: [17, 27, 22]

logging:
  level: "INFO"
  file: "/var/log/pentary.log"
EOF

echo ""
echo "Step 7: Setting up systemd service..."
cat > /etc/systemd/system/pentary.service << 'EOF'
[Unit]
Description=Pentary Processor Service
After=network.target

[Service]
Type=simple
User=pi
ExecStart=/usr/local/bin/pentary-sim --daemon
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable pentary.service

echo ""
echo "Step 8: Creating desktop shortcuts..."
if [ -d /home/pi/Desktop ]; then
    cat > /home/pi/Desktop/Pentary.desktop << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=Pentary Processor
Comment=Pentary Computing Environment
Exec=/usr/local/bin/pentary-demo
Icon=/opt/pentary/icon.png
Terminal=true
Categories=Development;Education;
EOF
    chmod +x /home/pi/Desktop/Pentary.desktop
    chown pi:pi /home/pi/Desktop/Pentary.desktop
fi

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Quick Start:"
echo "  pentary --version    # Check installation"
echo "  pentary-test         # Run tests"
echo "  pentary-demo         # Run demo"
echo ""
echo "Documentation:"
echo "  /opt/pentary/docs/"
echo ""
echo "Configuration:"
echo "  /etc/pentary/config.yaml"
echo ""
echo "Reboot recommended for all changes to take effect."
echo ""