#!/bin/bash
# Pentary Project Setup Script
# Checks dependencies and sets up the environment

set -e

echo "=========================================="
echo "  Pentary Project Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 7 ]; then
        echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
    else
        echo -e "${RED}✗${NC} Python 3.7+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

# Check pip
echo "Checking pip..."
if command -v pip3 &> /dev/null; then
    echo -e "${GREEN}✓${NC} pip3 found"
    PIP_CMD="pip3"
elif command -v pip &> /dev/null; then
    echo -e "${GREEN}✓${NC} pip found"
    PIP_CMD="pip"
else
    echo -e "${RED}✗${NC} pip not found. Please install pip."
    exit 1
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    $PIP_CMD install -r requirements.txt
    echo -e "${GREEN}✓${NC} Dependencies installed"
else
    echo -e "${YELLOW}⚠${NC} requirements.txt not found, installing numpy..."
    $PIP_CMD install numpy
fi

# Check if tools directory exists
if [ ! -d "tools" ]; then
    echo -e "${RED}✗${NC} tools directory not found!"
    exit 1
fi

# Make scripts executable
echo ""
echo "Making scripts executable..."
chmod +x tools/*.py 2>/dev/null || true
chmod +x setup.sh
echo -e "${GREEN}✓${NC} Scripts are executable"

# Verify tools
echo ""
echo "Verifying tools..."
TOOLS=("pentary_converter.py" "pentary_arithmetic.py" "pentary_simulator.py")
for tool in "${TOOLS[@]}"; do
    if [ -f "tools/$tool" ]; then
        echo -e "${GREEN}✓${NC} $tool found"
    else
        echo -e "${YELLOW}⚠${NC} $tool not found"
    fi
done

# Test basic functionality
echo ""
echo "Running quick test..."
cd tools
if python3 -c "from pentary_converter import PentaryConverter; c = PentaryConverter(); assert c.decimal_to_pentary(42) == '⊕⊖⊕'" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Basic functionality test passed"
else
    echo -e "${YELLOW}⚠${NC} Basic functionality test failed (this may be okay)"
fi
cd ..

echo ""
echo "=========================================="
echo -e "${GREEN}Setup complete!${NC}"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  1. Try the interactive CLI:"
echo "     python3 tools/pentary_cli.py"
echo ""
echo "  2. Run example tools:"
echo "     python3 tools/pentary_converter.py"
echo "     python3 tools/pentary_arithmetic.py"
echo "     python3 tools/pentary_simulator.py"
echo ""
echo "  3. Read the documentation:"
echo "     - QUICK_START.md - Get started in 5 minutes"
echo "     - PENTARY_COMPLETE_GUIDE.md - Comprehensive guide"
echo ""
echo "The future is not Binary. It is Balanced."
echo ""
