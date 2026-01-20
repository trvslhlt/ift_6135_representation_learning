# IFT 6135 Representation Learning

Course materials and notebooks for IFT 6135.

## Quick Start

### Option 1: Google Colab (No setup required)

Click the badge to open any notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/trvslhlt/ift_6135_representation_learning/blob/main/notebooks/00_setup_test.ipynb)

The first cell in each notebook handles cloning and dependency installation automatically.

### Option 2: Local Development

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone and setup**:
   ```bash
   git clone https://github.com/trvslhlt/ift_6135_representation_learning.git
   cd ift_6135_representation_learning
   uv sync
   ```

3. **Start JupyterLab**:
   ```bash
   uv run jupyter lab
   ```

## Deep Learning (Optional)

To install PyTorch and torchvision:

```bash
uv sync --extra deep
```

## Project Structure

```
ift_6135_representation_learning/
├── notebooks/          # Jupyter notebooks
├── assignments/        # Homework assignments
├── data/              # Datasets (small files only)
└── src/ift6135/       # Reusable Python utilities
```

## Adding Dependencies

```bash
# Add a new package
uv add package_name

# Add a development-only package
uv add --dev package_name
```
