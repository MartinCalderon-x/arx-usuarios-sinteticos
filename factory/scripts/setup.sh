#!/bin/bash
#
# Factory Setup Script
# Sets up the AI building tools (Arx-SDK) for product development.
#
set -e

echo "=================================="
echo "  Product Factory - Setup"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}Project root: $PROJECT_ROOT${NC}"
echo ""

# 1. Create virtual environment for factory (Python 3.13)
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment with Python 3.13...${NC}"
    python3.13 -m venv venv
    echo -e "${GREEN}Virtual environment created.${NC}"
else
    echo "Virtual environment already exists."
fi

# 2. Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# 3. Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# 4. Install factory dependencies (Arx-SDK)
echo -e "${YELLOW}Installing Arx-SDK...${NC}"
pip install -r factory/requirements.txt --quiet
echo -e "${GREEN}Arx-SDK installed.${NC}"

# 5. Create data directories
echo "Creating data directories..."
mkdir -p data/vectordb
mkdir -p data/cache
mkdir -p data/logs

# 6. Setup environment file
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${YELLOW}Created .env from .env.example${NC}"
        echo -e "${YELLOW}Please edit .env with your API keys.${NC}"
    fi
else
    echo ".env file already exists."
fi

echo ""
echo "=================================="
echo -e "${GREEN}  Factory Setup Complete!${NC}"
echo "=================================="
echo ""
echo "Your AI building tools are ready."
echo ""
echo "Next steps:"
echo "  1. Edit .env with your API keys"
echo "  2. Activate: source venv/bin/activate"
echo "  3. Test factory: python factory/run.py"
echo "  4. Create your product in product/"
echo ""
echo -e "${BLUE}Folder structure:${NC}"
echo "  factory/  - AI tools (Arx-SDK)"
echo "  product/  - Your product (any technology)"
echo "  data/     - Factory runtime data"
echo ""
