#!/bin/bash

# Usage:
#   Zero-Shot Inference:  ./task2.sh 1 2 <image_dir> <pretrained_model> <output_predictions> # 2_1.py
#   Prompt Tuning Strat 1: ./task2.sh 2 1 <dataset_root> <trained_model_output> # 2_2.py
#                        ./task2.sh 2 2 <image_dir> <trained_model_input> <output_predictions> # 2_train.py

if [ "$1" == "1" ] && [ "$2" == "2" ]; then
    # Zero-Shot Inference: Run 2_1.py (with image directory and pretrained model)
    python ./2_1.py "$4" "$5" "$6"

elif [ "$1" == "2" ] && [ "$2" == "1" ]; then
    # Prompt Tuning Strat 1: Run 2_2.py (with dataset root and output model)
    python ./2_2.py "$4" "$5"

elif [ "$1" == "2" ] && [ "$2" == "2" ]; then
    # Prompt Tuning Strat 2: Run 2_train.py (with image directory, pretrained model, and output predictions)
    python ./2_train.py "$4" "$5" "$6"

else
    echo "Invalid arguments"
    echo "See usage in script header."
    exit 1
fi
