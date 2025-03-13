#!/bin/bash

# Create results directory if it doesn't exist
mkdir -p results/mpgnn_simple

# Run the training script
echo "Starting SimpleMPGNN model training..."
python train_mpgnn.py

# Check if model was generated successfully
if [ -f "results/mpgnn_simple/best_model.pt" ]; then
    echo "Training completed successfully!"
    echo "Model saved to results/mpgnn_simple/best_model.pt"
    echo "Results saved to results/mpgnn_simple/"
else
    echo "Training failed - model file not found"
fi 