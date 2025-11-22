#!/usr/bin/sh

# Script to sequentially run training with all data efficiency strategy configs
# Usage: ./train_all_strategies.sh

set -e  # Exit on error

echo "=========================================="
echo "Data Efficiency Strategies Training Script"
echo "=========================================="

# 1. Check if uv is installed
echo ""
echo "Step 1: Checking environment..."
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install uv: https://github.com/astral-sh/uv"
    exit 1
fi

# 2. Sync environment
echo ""
echo "Step 2: Syncing environment..."
uv sync
echo "✓ Environment synced"

# 3. Check if dataset exists
echo ""
echo "Step 3: Checking dataset..."
if [ -d "./data/train" ] && [ -d "./data/validation" ] && [ -d "./data/test" ]; then
    echo "✓ Dataset already downloaded (skipping)"
else
    echo "Downloading dataset..."
    uv run download_dataset
    echo "✓ Dataset downloaded"
fi

# 4. Load environment variables
echo ""
echo "Step 4: Loading environment variables..."
if [ -f ".env" ]; then
    set -a  # Automatically export all variables
    source .env
    set +a
    echo "✓ Environment variables loaded from .env"
else
    echo "⚠ .env file not found. Continuing without it."
    echo "  For ClearML usage, create .env with CLEARML_API_ACCESS_KEY and CLEARML_API_SECRET_KEY"
fi

# 5. Define config files to run
CONFIGS=(
    "configs/train_config_edfs_lite.json"
    "configs/train_config_entropy_diversity.json"
    "configs/train_config_k_center.json"
    "configs/train_config_lexical_diversity.json"
    "configs/train_config_qdit_lite.json"
)

# 6. Run training for each config
echo ""
echo "Step 5: Running training for all strategies..."
echo "=========================================="

TOTAL=${#CONFIGS[@]}
CURRENT=0

for CONFIG_FILE in "${CONFIGS[@]}"; do
    CURRENT=$((CURRENT + 1))
    
    # Check if config exists
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "⚠ Warning: Config $CONFIG_FILE not found. Skipping..."
        continue
    fi
    
    # Extract strategy name from config file
    STRATEGY_NAME=$(basename "$CONFIG_FILE" .json | sed 's/train_config_//')
    
    echo ""
    echo "----------------------------------------"
    echo "[$CURRENT/$TOTAL] Running: $STRATEGY_NAME"
    echo "Config: $CONFIG_FILE"
    echo "----------------------------------------"
    
    # Run training
    uv run run --config "$CONFIG_FILE"
    
    echo ""
    echo "✓ Completed: $STRATEGY_NAME"
done

echo ""
echo "=========================================="
echo "All training runs completed!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Total configs: $TOTAL"
echo "  - Completed: $CURRENT"
echo ""
echo "Checkpoints saved in: ./checkpoints/"
echo "Logs saved in: ./runs/"
