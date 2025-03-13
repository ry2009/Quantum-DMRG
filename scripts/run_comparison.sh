#!/bin/bash

# Define directories
RESULTS_DIR="results"
COMPARISON_DIR="results/comparison"

# Create comparison directory if it doesn't exist
mkdir -p "$COMPARISON_DIR"

echo "Running model comparison..."

# Run model comparison with paper results included
python model_comparison.py \
    --base_dir "$RESULTS_DIR" \
    --include_paper \
    --output_dir "$COMPARISON_DIR"

# Check if comparison was successful
if [ $? -ne 0 ]; then
    echo "Error: Model comparison failed."
    exit 1
fi

echo "Model comparison completed successfully."
echo "Comparison results saved to $COMPARISON_DIR"

# Display the model comparison report
echo -e "\n==== MODEL COMPARISON REPORT ====\n"
cat "$COMPARISON_DIR/model_comparison_report.md"
echo -e "\n================================\n"

echo "Done!" 