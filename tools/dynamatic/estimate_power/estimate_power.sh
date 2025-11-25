#!/bin/bash

# Script arguments
OUTPUT_DIR=$1
KERNEL_NAME=$2
CP=$3

python tools/dynamatic/estimate_power/estimate_power.py --output_dir $OUTPUT_DIR --kernel_name $KERNEL_NAME --cp $CP