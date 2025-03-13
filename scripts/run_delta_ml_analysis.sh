#!/bin/bash

# Create output directories
mkdir -p dmrg_analysis_results
mkdir -p delta_ml_exploration
mkdir -p delta_ml_results

echo "========================================="
echo "DMRG Data Analysis and Δ-ML Implementation"
echo "========================================="

# Step 1: Run the data analysis to understand the dataset
echo -e "\n[1/4] Analyzing DMRG data structure and properties..."
python analyze_dmrg_data.py

# Step 2: Run detailed delta-ML exploration on specific systems
echo -e "\n[2/4] Exploring Δ-ML potential for selected systems..."
python delta_ml_exploration.py

# Step 3: Train and evaluate the Δ-ML model
echo -e "\n[3/4] Training and evaluating Δ-ML model..."
python delta_ml_model.py

# Step 4: Generate summary report
echo -e "\n[4/4] Generating summary report..."

# Combine results into a single report
cat > delta_ml_summary.md << EOF
# Δ-ML Analysis and Results Summary

## Dataset Overview
- Test systems: $(ls test | grep -v "processed\|raw\|.DS_Store\|sample" | wc -l) molecular systems
- Training systems: $(find train -type d -name "hc_c*" | wc -l) molecular configurations
- Bond dimensions: 256, 512, 768, 1024, 1536, 2048, 3072

## Data Structure Analysis
See detailed analysis in \`dmrg_analysis_results/\` directory.
- Each file contains HF energy, DMRG energy, truncation error, and orbital information
- Orbital graphs constructed with MI threshold = 0.004
- Graph properties vary with bond dimension and system size

## Δ-ML Model Performance
See detailed results in \`delta_ml_results/\` directory.
- Model predicts energy difference between M=256 and M=1024 calculations
- Used features: single-site entropy, orbital occupation, mutual information, truncation error
- Neural network: Message-passing GNN with global features and readout MLP

EOF

# Add model performance if results.json exists
if [ -f "delta_ml_results/results.json" ]; then
    echo "### Performance Metrics" >> delta_ml_summary.md
    echo "\`\`\`" >> delta_ml_summary.md
    cat delta_ml_results/results.json >> delta_ml_summary.md
    echo "\`\`\`" >> delta_ml_summary.md
fi

echo -e "\nAnalysis complete! Results are available in:"
echo "- dmrg_analysis_results/ - General data analysis"
echo "- delta_ml_exploration/ - Detailed Δ-ML exploration"
echo "- delta_ml_results/ - Model training and evaluation results"
echo "- delta_ml_summary.md - Summary report" 