#!/bin/bash

# Usage:
#   Evaluation:        ./task1.sh 1 0 2 <image_dir> <pretrained_model> <output_predictions>
#   Fine-Tune Strat 1: ./task1.sh 2 1 1 <dataset_root> <trained_model_output>
#                      ./task1.sh 2 1 2 <image_dir> <trained_model_input> <output_predictions>
#   Fine-Tune Strat 2: ./task1.sh 2 2 1 <dataset_root> <trained_model_output>
#                      ./task1.sh 2 2 2 <image_dir> <trained_model_input> <output_predictions>
#   Fine-Tune Strat 3: ./task1.sh 2 3 1 <dataset_root> <trained_model_output>
#                      ./task1.sh 2 3 2 <image_dir> <trained_model_input> <output_predictions>

if [ "$1" == "1" ] && [ "$2" == "0" ] && [ "$3" == "2" ]; then
    python ./1_1.py "$4" "$5" "$6"

elif [ "$1" == "2" ] && [ "$3" == "1" ]; then
    python ./1_train.py "$4" "$5"

elif [ "$1" == "2" ] && [ "$3" == "2" ]; then
    python ./1_2.py "$4" "$5" "$6"

else
    echo "Invalid arguments"
    echo "See usage in script header."
    exit 1
fi
