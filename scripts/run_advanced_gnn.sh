#!/bin/bash

# Directory settings
TRAIN_DIR="train"
TEST_DIR="test"
RESULTS_DIR="results/advanced_gnn"

# Model parameters
BOND_DIM=512
SYSTEM_TYPE="pah"
MI_THRESHOLD=0.01
HIDDEN_SIZE=256
NUM_LAYERS=5
ATTENTION_HEADS=8
DROPOUT=0.15
GATING=True
POOLING="combined"
READOUT_LAYERS=3

# Training parameters
BATCH_SIZE=4
LR=0.0001
WEIGHT_DECAY=5e-4
EPOCHS=200
PATIENCE=25
VAL_SPLIT=0.15
TEST_SPLIT=0.1
GRAD_CLIP=0.5
WARMUP_EPOCHS=10
SEED=42

# Uncertainty parameters
DROPOUT_SAMPLES=20  # Number of Monte Carlo dropout samples for uncertainty estimation

# Create results directory if it doesn't exist
if [ ! -d "$RESULTS_DIR" ]; then
    mkdir -p "$RESULTS_DIR"
    echo "Created results directory: $RESULTS_DIR"
fi

# Run training
echo "Starting advanced GNN training..."
python train_advanced_gnn.py \
    --data_dir "$TRAIN_DIR" \
    --bond_dim "$BOND_DIM" \
    --system_type "$SYSTEM_TYPE" \
    --mi_threshold "$MI_THRESHOLD" \
    --hidden_size "$HIDDEN_SIZE" \
    --num_layers "$NUM_LAYERS" \
    --attention_heads "$ATTENTION_HEADS" \
    --dropout "$DROPOUT" \
    --pooling "$POOLING" \
    --gating "$GATING" \
    --readout_layers "$READOUT_LAYERS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --val_split "$VAL_SPLIT" \
    --test_split "$TEST_SPLIT" \
    --grad_clip "$GRAD_CLIP" \
    --warmup_epochs "$WARMUP_EPOCHS" \
    --use_lr_scheduler \
    --lr_scheduler_type "cosine" \
    --seed "$SEED" \
    --output_dir "$RESULTS_DIR"

# Check if training was successful
MODEL_DIR=$(find "$RESULTS_DIR" -name "advanced_gnn_*" -type d | sort -r | head -n 1)
if [ -z "$MODEL_DIR" ]; then
    echo "Error: No model directory found after training."
    exit 1
fi

MODEL_FILE="$MODEL_DIR/model.pt"
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found at $MODEL_FILE"
    exit 1
fi

# Run predictions on test data with uncertainty quantification
echo "Running predictions on test data with uncertainty quantification..."
python predict_advanced.py \
    --data_dir "$TEST_DIR" \
    --bond_dim "$BOND_DIM" \
    --system_type "$SYSTEM_TYPE" \
    --mi_threshold "$MI_THRESHOLD" \
    --model_path "$MODEL_FILE" \
    --batch_size 1 \
    --dropout_samples "$DROPOUT_SAMPLES" \
    --output_dir "$MODEL_DIR/test_predictions_with_uncertainty"

# Also run standard predictions for comparison
echo "Running standard predictions (without uncertainty)..."
python predict_advanced.py \
    --data_dir "$TEST_DIR" \
    --bond_dim "$BOND_DIM" \
    --system_type "$SYSTEM_TYPE" \
    --mi_threshold "$MI_THRESHOLD" \
    --model_path "$MODEL_FILE" \
    --batch_size 4 \
    --dropout_samples 0 \
    --output_dir "$MODEL_DIR/test_predictions_standard"

echo "All done! Results saved to $MODEL_DIR" 