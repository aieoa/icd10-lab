#!/bin/bash

# Setup conda environment on HPC cluster
# Name of your desired environment and source environment
TARGET_ENV="env-icd10-lab"
SOURCE_ENV="gpulab"
REQUIREMENTS_FILE="requirements.yml"

# Determine if target env exists
conda_env_exists() {
    conda info --envs | awk '{print $1}' | grep -q "^${TARGET_ENV}$"
}

# If target env non-existing, create clone of gpu-env and store under target name
if conda_env_exists; then
    echo "Conda environment '$TARGET_ENV' exists. Activating..."
else
    echo "Conda environment '$TARGET_ENV' does NOT exist. Cloning from '$SOURCE_ENV'..."
    conda create --name "$TARGET_ENV" --clone "$SOURCE_ENV"
    echo "Updating environment with $REQUIREMENTS_FILE ..."
    conda env update --name "$TARGET_ENV" --file "$REQUIREMENTS_FILE" --prune
fi

echo "Activating '$TARGET_ENV'..."
# Use the correct activation for shell type
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$TARGET_ENV"
