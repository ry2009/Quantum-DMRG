#!/bin/bash

# Locate the best model
SOTA_DIR="results/sota_gnn"
TEST_DIR="test"
OUTPUT_DIR="results/paper_comparison"

# Make sure the output directory exists
mkdir -p "$OUTPUT_DIR"

# Find the most recent SOTA model
MODEL_PATH=$(find "$SOTA_DIR" -name "model.pt" | sort -r | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: No SOTA model found in $SOTA_DIR"
    
    # Try to find any model
    MODEL_PATH=$(find "results" -name "model.pt" | sort -r | head -n 1)
    
    if [ -z "$MODEL_PATH" ]; then
        echo "Error: No model found in results directory"
        exit 1
    fi
    
    echo "Using alternative model: $MODEL_PATH"
fi

echo "Running comparison against paper using model: $MODEL_PATH"

# Run the comparison
python test_sota_vs_paper.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$TEST_DIR" \
    --bond_dim 512 \
    --system_type "pah" \
    --mi_threshold 0.01 \
    --output_dir "$OUTPUT_DIR"

# Check if comparison was successful
if [ $? -ne 0 ]; then
    echo "Error: Comparison failed"
    exit 1
fi

echo "Comparison completed successfully. Results saved to $OUTPUT_DIR"

# Display a summary of the results if available
RESULTS_FILE="$OUTPUT_DIR/detailed_comparison.csv"
if [ -f "$RESULTS_FILE" ]; then
    echo -e "\n==== SUMMARY OF RESULTS ====\n"
    echo "Our model vs Paper comparison:"
    column -t -s, "$RESULTS_FILE" | head -1
    echo "-----------------------------------"
    column -t -s, "$RESULTS_FILE" | tail -n +2
    echo -e "\n===========================\n"
fi

echo "Done!" 