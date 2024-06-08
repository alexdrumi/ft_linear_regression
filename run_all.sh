#!/bin/bash

# Function to check if a package is installed
check_package() {
    python3 -c "import $1" &> /dev/null
    if [ $? -ne 0 ]; then
        return 1
    else
        return 0
    fi
}

# List of required packages
REQUIRED_PACKAGES=("pandas" "matplotlib")

# Install missing packages
for PACKAGE in "${REQUIRED_PACKAGES[@]}"
do
    check_package $PACKAGE
    if [ $? -ne 0 ]; then
        echo "Package $PACKAGE is not installed. Installing..."
        pip3 install $PACKAGE
        if [ $? -ne 0 ]; then
            echo "Failed to install $PACKAGE. Exiting."
            exit 1
        fi
    else
        echo "Package $PACKAGE is already installed."
    fi
done

# Move to the src directory
cd "$(dirname "$0")/src"

# Run the training script
echo "Starting the training process..."
python3 train_linear_regression.py

# Check if the training script exited successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    echo "Starting the prediction process..."

    # Run the prediction script
    python3 predict_linear_regression.py
else
    echo "Training failed. Exiting."
    exit 1
fi

