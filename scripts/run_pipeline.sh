#!/bin/bash

# AI Tale Model Training Pipeline
# This script runs the complete pipeline: data preparation, model training, and evaluation

set -e  # Exit on error

# Configuration
DATA_CONFIG="configs/data_config.yaml"
TRAIN_CONFIG="configs/finetune_config.yaml"
OUTPUT_DIR="models/storyteller-v1"
MODEL="gpt2-medium"
EVAL_OUTPUT_DIR="evaluation_results"

# Print header
echo "==================================================="
echo "  AI Tale Model Training Pipeline"
echo "  $(date)"
echo "==================================================="

# Create directories
mkdir -p data/raw data/processed $OUTPUT_DIR $EVAL_OUTPUT_DIR logs

# Step 1: Data preparation
echo ""
echo "Step 1: Preparing dataset..."
python src/data/prepare_dataset.py --config $DATA_CONFIG
if [ $? -ne 0 ]; then
    echo "Error in data preparation. Exiting."
    exit 1
fi
echo "Dataset preparation completed."

# Step 2: Model training
echo ""
echo "Step 2: Training model..."
python src/train/finetune.py --config $TRAIN_CONFIG --model $MODEL --output_dir $OUTPUT_DIR
if [ $? -ne 0 ]; then
    echo "Error in model training. Exiting."
    exit 1
fi
echo "Model training completed."

# Step 3: Model evaluation
echo ""
echo "Step 3: Evaluating model..."
python src/evaluate/evaluate_model.py \
    --model-path $OUTPUT_DIR \
    --test-dataset data/processed/fairy_tales_test.json \
    --output-dir $EVAL_OUTPUT_DIR \
    --config $TRAIN_CONFIG \
    --num-samples 10
if [ $? -ne 0 ]; then
    echo "Error in model evaluation. Exiting."
    exit 1
fi
echo "Model evaluation completed."

# Step 4: Export model for production (optional)
if [ "$1" == "--export" ]; then
    echo ""
    echo "Step 4: Exporting model for production..."
    EXPORT_DIR="${OUTPUT_DIR}_production"
    python scripts/export_model.py \
        --model-path $OUTPUT_DIR \
        --output-dir $EXPORT_DIR \
        --format pytorch \
        --half-precision
    if [ $? -ne 0 ]; then
        echo "Error in model export. Exiting."
        exit 1
    fi
    echo "Model exported to $EXPORT_DIR"
fi

# Print summary
echo ""
echo "==================================================="
echo "  Pipeline completed successfully!"
echo "  Model saved to: $OUTPUT_DIR"
echo "  Evaluation results saved to: $EVAL_OUTPUT_DIR"
if [ "$1" == "--export" ]; then
    echo "  Production model exported to: $EXPORT_DIR"
fi
echo "===================================================" 