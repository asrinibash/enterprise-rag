# Installation Guide

Complete installation instructions for Enterprise RAG Knowledge Base.

## Prerequisites

1. **Python 3.9+** - [Download Python](https://www.python.org/downloads/)
2. **uv package manager** - Install with:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

## Quick Installation (Recommended)

### Step 1: Get the Project

```bash
# Clone the repository (if from Git)
git clone <your-repo-url>
cd enterprise-rag

# OR create new directory
mkdir enterprise-rag
cd enterprise-rag
# Then copy all project files into this directory
```

### Step 2: Run Installation Script

```bash
# Make script executable (Linux/Mac/Git Bash)
chmod +x install.sh

# Run installation
bash install.sh
```

This will:
- ✅ Create virtual environment
- ✅ Install all dependencies from `requirements.txt`
- ✅ Create necessary directories
- ✅ Set up `.env` file

### Step 3: Activate Environment

**Git Bash (Windows):**
```bash
source .venv/Scripts/activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**Windows PowerShell:**
```powershell
.venv\Scripts\Activate.ps1
```

### Step 4: Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list | grep -E "fastapi|langchain|faiss"
```

You should see all required packages installed.

## Manual Installation

If the script doesn't work, follow these manual steps:

### 1. Create Virtual Environment

```bash
cd enterprise-rag
uv venv
```

### 2. Activate Virtual Environment

```bash
# Git Bash on Windows
source .venv/Scripts/activate

# Linux/Mac
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

# Windows Command Prompt
.venv\Scripts\activate.bat
```

Your prompt should change to show `(.venv)` or `(enterprise-rag)`.

### 3. Install Dependencies

**Option A: From requirements.txt (Recommended)**
```bash
pip install -r requirements.txt
```

**Option B: Using uv pip**
```bash
uv pip install -r requirements.txt
```

**Option C: Manual package installation**
```bash
pip install fastapi uvicorn[standard] python-multipart pydantic pydantic-settings \
    langchain langchain-core langchain-community langchain-text-splitters \
    pypdf python-docx unstructured \
    faiss-cpu sentence-transformers rank-bm25 \
    openai tiktoken numpy pandas tqdm python-dotenv httpx
```

### 4. Create Directory Structure

```bash
mkdir -p data/documents data/indexes data/metadata
mkdir -p scripts tests
```

### 5. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env file with your settings (optional)
nano .env  # or use any text editor
```

## Verifying Installation

### Check All Components

```bash
# 1. Check Python imports
python -c "import fastapi, langchain, faiss, sentence_transformers; print('✓ All imports successful')"

# 2. Check project structure
ls -la src/
ls -la data/

# 3. Check configuration
python -c "from src.config import settings; print(f'✓ Config loaded: {settings.API_TITLE}')"
```

### Test Basic Functionality

```bash
# Create test script
cat > test_install.py << 'EOF'
from src.config import settings
from src.ingestion.embedder import Embedder

print("Testing installation...")
print(f"✓ Config loaded: {settings.API_TITLE}")

embedder = Embedder()
print(f"✓ Embedder loaded: {embedder.model_name}")

test_text = "This is a test sentence."
embedding = embedder.embed_text(test_text)
print(f"✓ Embedding generated: shape {embedding.shape}")

print("\n✅ Installation verified successfully!")
EOF

python test_install.py
```

## Platform-Specific Notes

### Windows (Git Bash)

```bash
# Always use forward slashes for activation
source .venv/Scripts/activate

# Use uv run to avoid activation
uv run python -m src.main
```

### Windows (PowerShell)

```powershell
# May need to enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate
.venv\Scripts\Activate.ps1
```

### Linux/Mac

```bash
# Standard activation
source .venv/bin/activate

# If permission denied
chmod +x install.sh
```

### Docker (Alternative)

If you prefer Docker:

```bash
# Build image
docker build -t enterprise-rag .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data enterprise-rag
```

## Troubleshooting

### Issue: "uv: command not found"

**Solution:**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.cargo/bin:$PATH"

# Restart terminal
```

### Issue: "No module named 'xyz'"

**Solution:**
```bash
# Make sure venv is activated
source .venv/Scripts/activate  # or appropriate command

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Permission denied"

**Solution:**
```bash
# Make scripts executable
chmod +x install.sh
chmod +x scripts/*.py

# Or run with bash
bash install.sh
```

### Issue: FAISS installation fails

**Solution:**
```bash
# Try CPU version explicitly
pip install faiss-cpu==1.7.4

# For GPU (if you have CUDA)
pip uninstall faiss-cpu
pip install faiss-gpu
```

### Issue: Sentence-transformers is slow

**Solution:**
```bash
# It downloads ~90MB model on first run
# Just wait for the download to complete
# Model is cached in ~/.cache/torch/sentence_transformers/
```

### Issue: "ImportError: langchain.schema"

**Solution:**
```bash
# Install missing langchain packages
pip install langchain-core langchain-text-splitters
```

## Development Installation

For development with testing tools:

```bash
# Install with dev dependencies
pip install -r requirements.txt

# Install dev tools
pip install pytest pytest-asyncio black ruff ipython

# Install in editable mode
pip install -e .
```

## Updating Dependencies

```bash
# Activate environment
source .venv/Scripts/activate

# Update all packages
pip install --upgrade -r requirements.txt

# Or update specific package
pip install --upgrade fastapi
```

## Uninstallation

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv

# Remove cached data (optional)
rm -rf data/indexes/*
rm -rf data/metadata/*
```

## Next Steps

After successful installation:

1. **Add Documents**: Place PDFs, TXT files in `data/documents/`
2. **Ingest Data**: Run `python scripts/ingest_documents.py`
3. **Start Server**: Run `python -m src.main`
4. **Test API**: Visit http://localhost:8000/docs

## Getting Help

- Check logs in terminal for error messages
- Review `.env` settings
- Ensure all files from artifacts are in correct locations
- Check Python version: `python --version` (must be 3.9+)

## Recommended Directory Structure After Installation

```
enterprise-rag/
├── .venv/                    ← Virtual environment
├── src/                      ← Source code
├── scripts/                  ← Utility scripts
├── data/                     ← Data directory
│   ├── documents/           ← Your documents here
│   ├── indexes/             ← Generated indexes
│   └── metadata/            ← Metadata storage
├── requirements.txt          ← Dependencies
├── .env                      ← Configuration
├── pyproject.toml           ← Project metadata
└── README.md                ← Documentation
```

---

**Installation complete!** 🎉 Proceed to the Getting Started guide.