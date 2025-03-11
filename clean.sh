#!/bin/bash

# Define files to remove
files=("src/thetas.txt" "data/testing_data.csv" "data/training_data.csv")
VENV_DIR="venv"

# 1. Remove specific files
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "Removing $file"
        rm "$file"
    else
        echo "$file does not exist"
    fi
done

# 2. Deactivate the virtual environment if active
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Deactivating virtual environment..."
    deactivate
else
    echo "Virtual environment is not active."
fi

# 3. Remove the virtual environment directory
if [ -d "$VENV_DIR" ]; then
    echo "Removing virtual environment directory: $VENV_DIR"
    rm -rf "$VENV_DIR"
else
    echo "Virtual environment directory does not exist."
fi

echo "Cleanup complete."
