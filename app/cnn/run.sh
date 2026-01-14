##!/bin/bash
#
## SOM Quality Analyzer - Quick Start Script
## This script provides convenient commands to run the project
#
#set -e  # Exit on error
#
## Colors for output
#RED='\033[0;31m'
#GREEN='\033[0;32m'
#YELLOW='\033[1;33m'
#BLUE='\033[0;34m'
#NC='\033[0m' # No Color
#
## Print header
#echo -e "${BLUE}================================${NC}"
#echo -e "${BLUE}SOM Quality Analyzer${NC}"
#echo -e "${BLUE}================================${NC}"
#
## Check if virtual environment exists
#VENV_DIR="venv"
#PYTHON_CMD="python3"
#
## Function to check if we're in a virtual environment
#in_venv() {
#    if [ -n "$VIRTUAL_ENV" ]; then
#        return 0
#    else
#        return 1
#    fi
#}
#
## Function to activate venv if it exists
#use_venv() {
#    if [ -d "$VENV_DIR" ] && ! in_venv; then
#        echo -e "${YELLOW}Activating virtual environment...${NC}"
#        source "$VENV_DIR/bin/activate"
#        PYTHON_CMD="python"
#    elif in_venv; then
#        PYTHON_CMD="python"
#    fi
#}
#
## Function to print usage
#usage() {
#    echo ""
#    echo "Usage: ./run.sh [command]"
#    echo ""
#    echo "Commands:"
#    echo "  setup         - Create virtual environment and install dependencies"
#    echo "  validate      - Validate your data before processing"
#    echo "  prepare       - Prepare the dataset from results.csv"
#    echo "  train         - Train the model (standard architecture)"
#    echo "  train-lite    - Train the lightweight model"
#    echo "  predict       - Make predictions on a single image"
#    echo "  evaluate      - Evaluate model on test set"
#    echo "  tensorboard   - Launch TensorBoard"
#    echo "  clean         - Clean generated files (models, logs)"
#    echo "  clean-all     - Clean everything including venv"
#    echo "  help          - Show this help message"
#    echo ""
#    echo "Examples:"
#    echo "  ./run.sh setup"
#    echo "  ./run.sh prepare"
#    echo "  ./run.sh train"
#    echo "  ./run.sh predict data/raw_maps/example.png"
#    echo ""
#}
#
## Command: Setup environment
#setup() {
#    echo -e "${GREEN}Setting up Python environment...${NC}"
#
#    # Check Python version
#    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
#    echo -e "${BLUE}Using Python $PYTHON_VERSION${NC}"
#
#    # Create virtual environment if it doesn't exist
#    if [ ! -d "$VENV_DIR" ]; then
#        echo -e "${GREEN}Creating virtual environment...${NC}"
#        $PYTHON_CMD -m venv $VENV_DIR
#    else
#        echo -e "${YELLOW}Virtual environment already exists${NC}"
#    fi
#
#    # Activate virtual environment
#    echo -e "${GREEN}Activating virtual environment...${NC}"
#    source "$VENV_DIR/bin/activate"
#
#    # Upgrade pip
#    echo -e "${GREEN}Upgrading pip...${NC}"
#    python -m pip install --upgrade pip
#
#    # Install dependencies
#    echo -e "${GREEN}Installing dependencies...${NC}"
#    pip install -r requirements.txt
#
#    echo -e "${GREEN}Setup complete!${NC}"
#    echo -e "${YELLOW}To activate the environment manually, run: source venv/bin/activate${NC}"
#}
#
## Command: Validate data
#validate() {
#    use_venv
#    echo -e "${GREEN}Validating dataset...${NC}"
#    $PYTHON_CMD src/validate_data.py
#}
#
## Command: Prepare data
#prepare() {
#    use_venv
#    echo -e "${GREEN}Preparing dataset...${NC}"
#    $PYTHON_CMD src/prepare_data.py
#    echo -e "${GREEN}Dataset preparation complete!${NC}"
#}
#
## Command: Train (standard model)
#train() {
#    use_venv
#    echo -e "${GREEN}Training standard model...${NC}"
#    $PYTHON_CMD src/train.py \
#        --model standard \
#        --epochs 50 \
#        --batch-size 32
#    echo -e "${GREEN}Training complete!${NC}"
#}
#
## Command: Train lightweight model
#train_lite() {
#    use_venv
#    echo -e "${GREEN}Training lightweight model...${NC}"
#    $PYTHON_CMD src/train.py \
#        --model lite \
#        --epochs 50 \
#        --batch-size 32
#    echo -e "${GREEN}Training complete!${NC}"
#}
#
## Command: Predict
#predict() {
#    use_venv
#    if [ -z "$1" ]; then
#        echo -e "${RED}Error: Please provide an image path${NC}"
#        echo "Usage: ./run.sh predict <image_path>"
#        exit 1
#    fi
#
#    # Find the latest model
#    MODEL=$(ls -t models/*_best.keras 2>/dev/null | head -1)
#
#    if [ -z "$MODEL" ]; then
#        echo -e "${RED}Error: No trained model found in models/${NC}"
#        echo "Please train a model first using: ./run.sh train"
#        exit 1
#    fi
#
#    echo -e "${GREEN}Using model: $MODEL${NC}"
#    echo -e "${GREEN}Predicting quality for: $1${NC}"
#    $PYTHON_CMD src/predict.py \
#        --model "$MODEL" \
#        --image "$1"
#}
#
## Command: Evaluate
#evaluate() {
#    use_venv
#    # Find the latest model and test set
#    MODEL=$(ls -t models/*_best.keras 2>/dev/null | head -1)
#    TEST_CSV=$(ls -t models/*_test_set.csv 2>/dev/null | head -1)
#
#    if [ -z "$MODEL" ] || [ -z "$TEST_CSV" ]; then
#        echo -e "${RED}Error: Model or test set not found${NC}"
#        echo "Please train a model first using: ./run.sh train"
#        exit 1
#    fi
#
#    echo -e "${GREEN}Evaluating model: $MODEL${NC}"
#    echo -e "${GREEN}Using test set: $TEST_CSV${NC}"
#    $PYTHON_CMD src/evaluate.py \
#        --model "$MODEL" \
#        --test-csv "$TEST_CSV"
#    echo -e "${GREEN}Evaluation complete! Check evaluation_results/ directory${NC}"
#}
#
## Command: TensorBoard
#tensorboard() {
#    use_venv
#    # Find the latest log directory
#    LOG_DIR=$(ls -td logs/som_quality_* 2>/dev/null | head -1)
#
#    if [ -z "$LOG_DIR" ]; then
#        echo -e "${RED}Error: No training logs found${NC}"
#        echo "Please train a model first using: ./run.sh train"
#        exit 1
#    fi
#
#    echo -e "${GREEN}Launching TensorBoard for: $LOG_DIR${NC}"
#    echo -e "${YELLOW}Open your browser to: http://localhost:6006${NC}"
#    echo -e "${YELLOW}Press Ctrl+C to stop TensorBoard${NC}"
#    $PYTHON_CMD -m tensorboard.main --logdir="$LOG_DIR" --host=0.0.0.0 --port=6006
#}
#
## Command: Clean
#clean() {
#    echo -e "${YELLOW}Warning: This will delete all trained models and logs!${NC}"
#    read -p "Are you sure? (y/N) " -n 1 -r
#    echo
#    if [[ $REPLY =~ ^[Yy]$ ]]; then
#        echo -e "${GREEN}Cleaning generated files...${NC}"
#        rm -rf models/*.keras models/*.h5 models/*.csv
#        rm -rf logs/*
#        rm -rf evaluation_results/
#        rm -f predictions.csv
#        echo -e "${GREEN}Clean complete!${NC}"
#    else
#        echo -e "${YELLOW}Clean cancelled${NC}"
#    fi
#}
#
## Command: Clean all (including venv)
#clean_all() {
#    echo -e "${YELLOW}Warning: This will delete models, logs, AND the virtual environment!${NC}"
#    read -p "Are you sure? (y/N) " -n 1 -r
#    echo
#    if [[ $REPLY =~ ^[Yy]$ ]]; then
#        echo -e "${GREEN}Cleaning all files...${NC}"
#        rm -rf models/*.keras models/*.h5 models/*.csv
#        rm -rf logs/*
#        rm -rf evaluation_results/
#        rm -f predictions.csv
#        rm -rf $VENV_DIR
#        echo -e "${GREEN}Clean complete!${NC}"
#    else
#        echo -e "${YELLOW}Clean cancelled${NC}"
#    fi
#}
#
## Main script logic
#case "$1" in
#    setup)
#        setup
#        ;;
#    validate)
#        validate
#        ;;
#    prepare)
#        prepare
#        ;;
#    train)
#        train
#        ;;
#    train-lite)
#        train_lite
#        ;;
#    predict)
#        predict "$2"
#        ;;
#    evaluate)
#        evaluate
#        ;;
#    tensorboard)
#        tensorboard
#        ;;
#    clean)
#        clean
#        ;;
#    clean-all)
#        clean_all
#        ;;
#    help|"")
#        usage
#        ;;
#    *)
#        echo -e "${RED}Unknown command: $1${NC}"
#        usage
#        exit 1
#        ;;
#esac
