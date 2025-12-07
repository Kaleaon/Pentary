# Pentary Processor USB Drive Prototype

## Overview

The Pentary USB Drive Prototype is a portable, plug-and-play implementation that runs directly from a USB drive on any computer. No installation required - just plug in and run!

## Features

- ✅ **Plug-and-Play:** Works on Windows, macOS, and Linux
- ✅ **Portable:** Carry your Pentary environment anywhere
- ✅ **No Installation:** Runs entirely from USB drive
- ✅ **Persistent Storage:** Save your work on the drive
- ✅ **Pre-configured:** All tools and demos included
- ✅ **Bootable Option:** Can boot as live OS

---

## Quick Start (2 Minutes)

### Step 1: Download Image

Download the pre-built USB image:
- **Standard Version (4 GB):** https://pentary.org/downloads/pentary-usb-standard.img
- **Full Version (8 GB):** https://pentary.org/downloads/pentary-usb-full.img
- **Bootable Version (16 GB):** https://pentary.org/downloads/pentary-usb-bootable.img

### Step 2: Write to USB Drive

#### Windows
```powershell
# Using Rufus (recommended)
1. Download Rufus: https://rufus.ie
2. Select USB drive
3. Select pentary-usb.img
4. Click START
```

#### macOS
```bash
# Using dd
diskutil list
diskutil unmountDisk /dev/diskN
sudo dd if=pentary-usb.img of=/dev/rdiskN bs=1m
```

#### Linux
```bash
# Using dd
lsblk
sudo dd if=pentary-usb.img of=/dev/sdX bs=4M status=progress
sync
```

### Step 3: Run Pentary

1. Plug in USB drive
2. Open file browser
3. Double-click `Pentary-Launcher`
4. Start using Pentary!

---

## USB Drive Versions

### Standard Version (4 GB)

**Contents:**
- Pentary emulator (Python)
- Basic tools and utilities
- Demo applications
- Documentation
- Sample programs

**Requirements:**
- 4 GB USB drive (8 GB recommended)
- Any OS with Python 3.8+
- 2 GB RAM minimum

**Use Cases:**
- Learning pentary computing
- Running demos
- Basic development

### Full Version (8 GB)

**Contents:**
- Everything in Standard
- Pre-trained neural network models
- Complete development environment
- Video tutorials
- Extended documentation
- Benchmark suite

**Requirements:**
- 8 GB USB drive (16 GB recommended)
- Any OS with Python 3.8+
- 4 GB RAM minimum

**Use Cases:**
- Full development
- Neural network experiments
- Performance testing
- Teaching and presentations

### Bootable Version (16 GB)

**Contents:**
- Everything in Full
- Bootable Linux OS (Ubuntu-based)
- Pre-installed development tools
- GPU acceleration support
- Hardware interface tools

**Requirements:**
- 16 GB USB drive (32 GB recommended)
- UEFI-capable computer
- 4 GB RAM minimum

**Use Cases:**
- Dedicated pentary workstation
- Hardware testing
- Maximum performance
- Isolated environment

---

## Directory Structure

```
USB Drive Root
├── Pentary-Launcher.exe        # Windows launcher
├── Pentary-Launcher.app        # macOS launcher
├── Pentary-Launcher.sh         # Linux launcher
├── README.txt                  # Quick start guide
│
├── pentary/                    # Core software
│   ├── bin/                    # Executables
│   ├── lib/                    # Libraries
│   ├── tools/                  # Development tools
│   └── python/                 # Python environment
│
├── demos/                      # Demo applications
│   ├── calculator/
│   ├── neural-network/
│   ├── image-processing/
│   └── benchmarks/
│
├── models/                     # Pre-trained models
│   ├── mnist/
│   ├── cifar10/
│   └── resnet/
│
├── docs/                       # Documentation
│   ├── getting-started.pdf
│   ├── api-reference.pdf
│   └── tutorials/
│
├── examples/                   # Example programs
│   ├── basic/
│   ├── intermediate/
│   └── advanced/
│
├── workspace/                  # User workspace
│   ├── projects/
│   ├── data/
│   └── results/
│
└── config/                     # Configuration
    ├── settings.yaml
    └── preferences.json
```

---

## Platform-Specific Instructions

### Windows

#### Running from USB

1. Plug in USB drive
2. Open USB drive in Explorer
3. Double-click `Pentary-Launcher.exe`
4. Choose option:
   - **Quick Demo:** Run interactive demo
   - **Development:** Open development environment
   - **Benchmarks:** Run performance tests

#### Command Line

```cmd
# Navigate to USB drive
D:
cd pentary

# Run pentary
python\python.exe bin\pentary.py

# Run demo
python\python.exe demos\calculator\calc.py
```

#### Troubleshooting Windows

**Issue:** "Windows protected your PC"
- Click "More info"
- Click "Run anyway"

**Issue:** Python not found
- Use included Python: `python\python.exe`

**Issue:** Permission denied
- Run as Administrator
- Or copy to local drive

### macOS

#### Running from USB

1. Plug in USB drive
2. Open USB drive in Finder
3. Double-click `Pentary-Launcher.app`
4. If blocked by Gatekeeper:
   - Right-click → Open
   - Click "Open" in dialog

#### Command Line

```bash
# Navigate to USB drive
cd /Volumes/PENTARY

# Run pentary
./pentary/python/bin/python3 pentary/bin/pentary.py

# Run demo
./pentary/python/bin/python3 demos/calculator/calc.py
```

#### Troubleshooting macOS

**Issue:** "App can't be opened"
- System Preferences → Security & Privacy
- Click "Open Anyway"

**Issue:** Slow performance
- Copy to local drive for better speed
- USB 3.0 recommended

### Linux

#### Running from USB

1. Plug in USB drive
2. Mount automatically or manually:
   ```bash
   sudo mount /dev/sdb1 /mnt/pentary
   ```
3. Run launcher:
   ```bash
   cd /mnt/pentary
   ./Pentary-Launcher.sh
   ```

#### Command Line

```bash
# Navigate to USB drive
cd /media/$USER/PENTARY

# Run pentary
./pentary/python/bin/python3 pentary/bin/pentary.py

# Run demo
./pentary/python/bin/python3 demos/calculator/calc.py
```

#### Troubleshooting Linux

**Issue:** Permission denied
```bash
chmod +x Pentary-Launcher.sh
chmod +x pentary/bin/*
```

**Issue:** USB not mounting
```bash
sudo fdisk -l
sudo mount -t vfat /dev/sdb1 /mnt/pentary
```

---

## Bootable USB Usage

### Booting from USB

1. Insert USB drive
2. Restart computer
3. Enter BIOS/UEFI (usually F2, F12, or DEL)
4. Select USB drive as boot device
5. Boot into Pentary OS

### Pentary OS Features

- **Based on:** Ubuntu 22.04 LTS
- **Desktop:** XFCE (lightweight)
- **Pre-installed:**
  - Python 3.11
  - Pentary tools
  - Development environment
  - GPU drivers (NVIDIA, AMD)
  - Documentation

### Persistent Storage

The bootable version includes persistent storage:
- Changes are saved to USB drive
- Install additional software
- Save projects and data
- Customize environment

### Performance Mode

Enable performance mode for maximum speed:

```bash
# Switch to performance governor
sudo cpupower frequency-set -g performance

# Disable power saving
sudo systemctl mask sleep.target suspend.target

# Optimize for USB
sudo hdparm -W1 /dev/sdb
```

---

## Creating Custom USB Drive

### Build Your Own

```bash
# Clone repository
git clone https://github.com/Kaleaon/Pentary.git
cd Pentary

# Run USB builder
sudo ./scripts/build_usb.sh

# Options:
#   --size 4G|8G|16G
#   --bootable
#   --include-models
#   --include-docs

# Example: Full bootable version
sudo ./scripts/build_usb.sh --size 16G --bootable --include-models --include-docs
```

### Customization Options

Edit `config/usb_build.yaml`:

```yaml
# USB Build Configuration

version: "1.0"
size: "8G"
bootable: false

include:
  - core: true
  - demos: true
  - models: true
  - docs: true
  - videos: false
  - source: false

models:
  - mnist
  - cifar10
  - resnet18

tools:
  - calculator
  - visualizer
  - profiler
  - debugger

python_version: "3.11"
```

---

## Demo Applications

### 1. Interactive Calculator

```bash
# Windows
python\python.exe demos\calculator\calc.py

# macOS/Linux
./pentary/python/bin/python3 demos/calculator/calc.py
```

**Features:**
- Pentary arithmetic operations
- Binary ↔ Pentary conversion
- Step-by-step visualization
- Export results

### 2. Neural Network Demo

```bash
# Run MNIST demo
python demos/neural-network/mnist_demo.py
```

**Features:**
- Handwritten digit recognition
- Real-time inference
- Performance comparison
- Model visualization

### 3. Image Processing

```bash
# Run image processing demo
python demos/image-processing/process.py input.jpg
```

**Features:**
- Pentary quantization
- Edge detection
- Filters and effects
- Batch processing

### 4. Benchmark Suite

```bash
# Run full benchmark
python demos/benchmarks/run_all.py
```

**Tests:**
- Arithmetic operations
- Memory bandwidth
- Neural network inference
- Comparison with FP32

---

## API Quick Reference

### Basic Usage

```python
from pentary import PentaryProcessor

# Create processor
cpu = PentaryProcessor()

# Set registers
cpu.set_register(0, 5)
cpu.set_register(1, 3)

# Execute instruction
cpu.execute("ADD R2, R0, R1")

# Get result
result = cpu.get_register(2)
print(f"Result: {result}")
```

### Neural Network

```python
from pentary import PentaryQuantizer
import numpy as np

# Create quantizer
quantizer = PentaryQuantizer()

# Load model
model = quantizer.load_model("models/mnist/model.pqm")

# Run inference
image = np.random.rand(1, 28, 28, 1)
prediction = model.predict(image)

print(f"Prediction: {np.argmax(prediction)}")
```

---

## Performance Optimization

### USB Speed Tips

1. **Use USB 3.0 or higher**
   - USB 2.0: ~30 MB/s
   - USB 3.0: ~100 MB/s
   - USB 3.1: ~300 MB/s

2. **Copy to RAM disk for speed**
   ```bash
   # Linux
   sudo mkdir /mnt/ramdisk
   sudo mount -t tmpfs -o size=4G tmpfs /mnt/ramdisk
   cp -r /media/PENTARY/* /mnt/ramdisk/
   cd /mnt/ramdisk
   ```

3. **Use SSD-based USB drive**
   - Much faster than flash drives
   - Better for bootable version

### Memory Optimization

```python
# Enable memory optimization
from pentary import PentaryConfig

config = PentaryConfig()
config.set('memory.optimize', True)
config.set('memory.cache_size', '256M')
config.save()
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Slow Performance

**Symptoms:**
- Long load times
- Slow execution
- High disk activity

**Solutions:**
1. Use USB 3.0 port
2. Copy to local drive
3. Use RAM disk
4. Upgrade to SSD USB drive

#### Issue 2: Permission Errors

**Symptoms:**
- "Access denied"
- "Permission denied"
- Can't write files

**Solutions:**
```bash
# Linux/macOS
chmod -R 755 /media/PENTARY
chmod +x pentary/bin/*

# Windows
# Run as Administrator
```

#### Issue 3: Python Not Found

**Symptoms:**
- "Python not found"
- "Module not found"

**Solutions:**
1. Use included Python:
   - Windows: `python\python.exe`
   - macOS/Linux: `./pentary/python/bin/python3`

2. Set PYTHONPATH:
   ```bash
   export PYTHONPATH=/media/PENTARY/pentary/lib
   ```

#### Issue 4: USB Not Detected

**Symptoms:**
- Drive not showing up
- Can't access files

**Solutions:**
```bash
# Linux
sudo fdisk -l
sudo mount /dev/sdb1 /mnt/pentary

# macOS
diskutil list
diskutil mount /dev/disk2s1

# Windows
# Check Disk Management
# Assign drive letter if needed
```

---

## Security Considerations

### Data Protection

1. **Encrypt USB drive**
   ```bash
   # Linux (LUKS)
   sudo cryptsetup luksFormat /dev/sdb1
   sudo cryptsetup open /dev/sdb1 pentary_encrypted
   ```

2. **Use read-only mode**
   - Protect against malware
   - Prevent accidental changes

3. **Backup regularly**
   ```bash
   # Backup workspace
   cp -r workspace/ ~/pentary_backup/
   ```

### Safe Usage

- Don't run on untrusted computers
- Scan for malware regularly
- Keep USB drive physically secure
- Use encryption for sensitive data

---

## Updates and Maintenance

### Updating Software

```bash
# Check for updates
python pentary/bin/update.py --check

# Download updates
python pentary/bin/update.py --download

# Install updates
python pentary/bin/update.py --install
```

### Cleaning Up

```bash
# Remove temporary files
python pentary/bin/cleanup.py

# Clear cache
rm -rf workspace/.cache/*

# Reset to defaults
python pentary/bin/reset.py
```

---

## Advanced Features

### Portable Development Environment

The USB drive includes a complete development environment:

```bash
# Start development environment
./pentary/bin/dev-env.sh

# Features:
# - Code editor (VS Code Portable)
# - Python IDE
# - Git client
# - Documentation browser
```

### Remote Access

Enable remote access to your USB Pentary:

```python
from pentary import PentaryServer

# Start server
server = PentaryServer(port=8080)
server.start()

# Access from browser
# http://localhost:8080
```

### Batch Processing

Process multiple files:

```bash
# Batch image processing
python pentary/bin/batch.py \
    --input images/*.jpg \
    --output processed/ \
    --operation quantize
```

---

## Educational Use

### Classroom Setup

1. **Prepare USB drives**
   - One per student
   - Pre-loaded with materials

2. **Distribute drives**
   - Students plug in and start
   - No installation needed

3. **Collect work**
   - Students save to workspace/
   - Collect USB drives

### Teaching Materials

Included on USB drive:
- Interactive tutorials
- Video lessons
- Exercises and solutions
- Quizzes and tests

---

## Resources

### Documentation
- Quick Start Guide: `docs/quick-start.pdf`
- Full Manual: `docs/manual.pdf`
- API Reference: `docs/api-reference.pdf`

### Videos
- Getting Started: `docs/videos/getting-started.mp4`
- Tutorials: `docs/videos/tutorials/`
- Demos: `docs/videos/demos/`

### Support
- Email: support@pentary.org
- Forum: https://forum.pentary.org
- GitHub: https://github.com/Kaleaon/Pentary

---

## Appendix

### USB Drive Specifications

**Recommended USB Drives:**

| Brand | Model | Size | Speed | Price |
|-------|-------|------|-------|-------|
| SanDisk | Extreme Pro | 128 GB | 420 MB/s | $30 |
| Samsung | BAR Plus | 64 GB | 300 MB/s | $15 |
| Kingston | DataTraveler | 32 GB | 200 MB/s | $10 |

### File Sizes

| Component | Size |
|-----------|------|
| Core software | 500 MB |
| Python environment | 300 MB |
| Demos | 200 MB |
| Models (MNIST) | 50 MB |
| Models (ResNet) | 500 MB |
| Documentation | 100 MB |
| Videos | 1 GB |

### System Requirements

**Minimum:**
- USB 2.0 port
- 2 GB RAM
- Any OS with Python 3.8+

**Recommended:**
- USB 3.0 port
- 4 GB RAM
- Modern OS (Windows 10+, macOS 10.15+, Ubuntu 20.04+)

**Optimal:**
- USB 3.1 port
- 8 GB RAM
- SSD-based USB drive
- GPU for acceleration

---

**Document Version:** 1.0  
**Last Updated:** December 2024  
**Status:** Ready for deployment  
**Tested On:** Windows 10/11, macOS 12+, Ubuntu 20.04+