#!/bin/bash

# Run the training script
echo "Starting the training process..."
python3 train_linear_regression.py

# Check if the training script exited successfully
if [ $? -eq 0 ]; then
    echo "Training completed successfully."
    echo "Starting the prediction process..."
    
    # Run the prediction script
    python3 predict_ft_linear_regression.py
else
    echo "Training failed. Exiting."
fi

