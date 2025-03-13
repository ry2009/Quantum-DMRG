#!/bin/bash

# Set variables
DATA_DIR="data/spin_systems"
RESULTS_DIR="results/bayesian_gnn"
SYSTEM_TYPE="heisenberg"
DIM=1
SIZE=10
HIDDEN_SIZE=64
NUM_LAYERS=3
NUM_SAMPLES=20
BATCH_SIZE=32
LR=0.001
WEIGHT_DECAY=1e-5
EPOCHS=100
PATIENCE=15

# Create data directory
mkdir -p $DATA_DIR

# Create results directory
mkdir -p $RESULTS_DIR

# Train the model
echo "Training Bayesian GNN model..."
python train_bayesian.py \
    --data_dir $DATA_DIR \
    --system_type $SYSTEM_TYPE \
    --dim $DIM \
    --size $SIZE \
    --hidden_size $HIDDEN_SIZE \
    --num_layers $NUM_LAYERS \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --epochs $EPOCHS \
    --patience $PATIENCE \
    --output_dir $RESULTS_DIR

# Get the most recent model directory
MODEL_DIR=$(ls -td $RESULTS_DIR/bayesian_gnn_* | head -1)
MODEL_PATH="$MODEL_DIR/model.pt"

# Run prediction
echo "Running prediction with uncertainty estimation..."
python predict.py \
    --model_path $MODEL_PATH \
    --data_dir $DATA_DIR \
    --system_type $SYSTEM_TYPE \
    --dim $DIM \
    --size $SIZE \
    --num_samples 50 \
    --output_dir "$MODEL_DIR/predictions"

echo "Done! Results are saved in $MODEL_DIR" 