# ml101

Machine Learning 101 - Educational notebooks for learning data science and machine learning fundamentals.

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/karolzak/ml101.git
cd ml101
```

### 2. Open in Dev Container (Recommended)

This project includes a dev container configuration for a consistent development environment:

1. **Prerequisites:**
   - Install [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Install [VS Code](https://code.visualstudio.com/)
   - Install the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

2. **Open the project:**
   - Open the project folder in VS Code
   - When prompted, click "Reopen in Container"
   - Or use Command Palette (F1) → "Dev Containers: Reopen in Container"

3. The dev container will automatically:
   - Set up Python 3.11 environment
   - Install all dependencies via `uv`
   - Configure Jupyter and required tools

### 3. Install Dependencies (Alternative: Local Setup)

If not using dev containers, install dependencies locally using [uv](https://github.com/astral-sh/uv):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync
```

### 4. Configure Environment Variables (Optional)

Some notebooks (e.g., `04-llm-basics.ipynb`) require API credentials for live demonstrations.

**Setup:**

1. Copy the example environment file:

   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your credentials:

   ```bash
   # Azure OpenAI Configuration
   AZURE_OPENAI_API_KEY=your-api-key-here
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   ```

3. **Important:** The `.env` file is gitignored and will never be committed to version control.

**Note:** If you don't configure these credentials, affected notebooks will gracefully fall back to simulated data for demonstrations.

## 📚 Notebooks

- **00-basics.ipynb** - Python and data science fundamentals
- **01-boston_housing.ipynb** - Regression and feature engineering
- **02-embeddings.ipynb** - Vector embeddings and semantic search
- **03-deep-learning.ipynb** - Neural networks with PyTorch
- **04-llm-basics.ipynb** - Understanding Large Language Models (Executive-level)

## 👤 Author

**Karol Zak** - [karolzak](https://github.com/karolzak)
