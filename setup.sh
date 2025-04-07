#!/bin/bash

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Healthcare Diagnostic Assistant...${NC}"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}Node.js is not installed. Please install Node.js v18 or higher.${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}npm is not installed. Please install npm.${NC}"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.9 or higher.${NC}"
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}pip3 is not installed. Please install pip3.${NC}"
    exit 1
fi

# Create frontend directory if it doesn't exist
if [ ! -d "frontend" ]; then
    echo -e "${YELLOW}Creating frontend directory...${NC}"
    mkdir -p frontend
fi

# Create backend directory if it doesn't exist
if [ ! -d "backend" ]; then
    echo -e "${YELLOW}Creating backend directory...${NC}"
    mkdir -p backend
fi

# Install frontend dependencies
echo -e "${GREEN}Installing frontend dependencies...${NC}"
cd frontend
if [ -f "package.json" ]; then
    npm install
else
    echo -e "${YELLOW}package.json not found in frontend directory. Skipping frontend installation.${NC}"
fi
cd ..

# Install backend dependencies
echo -e "${GREEN}Installing backend dependencies...${NC}"
cd backend
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
else
    echo -e "${YELLOW}requirements.txt not found in backend directory. Skipping backend installation.${NC}"
fi
cd ..

# Create .env files if they don't exist
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}Creating backend .env file...${NC}"
    cat > backend/.env << EOL
DATABASE_URL=postgresql://username:password@localhost:5432/healthcare
SECRET_KEY=your_secret_key
EOL
    echo -e "${YELLOW}Please update backend/.env with your database credentials and secret key.${NC}"
fi

if [ ! -f "frontend/.env.local" ]; then
    echo -e "${YELLOW}Creating frontend .env.local file...${NC}"
    cat > frontend/.env.local << EOL
NEXT_PUBLIC_API_URL=http://localhost:8000
EOL
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}To start the development servers:${NC}"
echo -e "1. Start the backend server: ${GREEN}cd backend && uvicorn app.main:app --reload${NC}"
echo -e "2. Start the frontend server: ${GREEN}cd frontend && npm run dev${NC}"
echo -e "3. Open your browser and navigate to ${GREEN}http://localhost:3000${NC}" 