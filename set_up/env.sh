#!/bin/bash

# =============================================================================
# Script Name: setup_env.sh
# Description: Creates a Conda environment and installs required packages
#              for the training scripts.
# Usage:        Run `bash setup_env.sh` in your terminal.
# =============================================================================

# ------------------------------
# 1. Define Variables
# ------------------------------

# Name of the Conda environment
ENV_NAME="cell_cls"

# Specify the Python version you want to use
PYTHON_VERSION="3.9"

# Required Conda packages with specific versions (optional)
# You can modify the versions based on your compatibility needs
CONDA_PACKAGES=(
    "pandas"                   # Data manipulation
    "numpy"                    # Numerical computations
    "scikit-learn"             # Machine learning utilities
    "matplotlib"               # Plotting
    "seaborn"                 # Statistical data visualization
    "tqdm"                    # Progress bars
    "tensorboard"             # TensorBoard for visualization
    "pillow"                   # Image processing
)

# Required pip packages with specific versions (optional)
# You can modify the versions based on your compatibility needs
PIP_PACKAGES=(
    "timm"                   # PyTorch Image Models
    "albumentations"          # Advanced data augmentation (optional)
    "wandb"
)

# ------------------------------
# 2. Check for Conda Installation
# ------------------------------

if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Conda before running this script."
    exit 1
fi

# ------------------------------
# 3. Create Conda Environment
# ------------------------------

# Check if the Conda environment already exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create --name "$ENV_NAME" python="$PYTHON_VERSION" -y

    if [ $? -ne 0 ]; then
        echo "Failed to create Conda environment '$ENV_NAME'. Exiting."
        exit 1
    fi

    echo "Conda environment '$ENV_NAME' created successfully."
fi

# ------------------------------
# 4. Activate Conda Environment
# ------------------------------

# Activate the Conda environment
echo "Activating Conda environment '$ENV_NAME'..."
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$ENV_NAME"

if [ $? -ne 0 ]; then
    echo "Failed to activate Conda environment '$ENV_NAME'. Exiting."
    exit 1
fi

echo "Conda environment '$ENV_NAME' activated."

# ------------------------------
# 5. Upgrade pip
# ------------------------------

echo "Upgrading pip..."
pip install --upgrade pip

if [ $? -ne 0 ]; then
    echo "Failed to upgrade pip. Exiting."
    conda deactivate
    exit 1
fi

echo "pip upgraded successfully."

# ------------------------------
# 6. Install Required Conda Packages
# ------------------------------

echo "Installing required Conda packages..."

for package in "${CONDA_PACKAGES[@]}"; do
    echo "Installing $package..."
    conda install "$package" -y

    if [ $? -ne 0 ]; then
        echo "Failed to install Conda package '$package'. Exiting."
        conda deactivate
        exit 1
    fi
done

echo "All required Conda packages installed successfully."

# ------------------------------
# 7. Install Required pip Packages
# ------------------------------

echo "Installing required pip packages..."

for package in "${PIP_PACKAGES[@]}"; do
    echo "Installing $package..."
    pip install "$package"

    if [ $? -ne 0 ]; then
        echo "Failed to install pip package '$package'. Exiting."
        conda deactivate
        exit 1
    fi
done

echo "All required pip packages installed successfully."

# ------------------------------
# 8. Deactivate Conda Environment (Optional)
# ------------------------------

# Uncomment the following line if you want to deactivate the environment after setup
# conda deactivate

# ------------------------------
# 9. Final Messages
# ------------------------------

echo "Setup completed successfully!"
echo "To activate the Conda environment in the future, run:"
echo "    conda activate $ENV_NAME"
echo "To deactivate the Conda environment, run:"
echo "    conda deactivate"