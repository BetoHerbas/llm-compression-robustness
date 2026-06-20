#!/bin/bash
cd /home/ubuntu/tesis_experiments
source venv/bin/activate

export COMPRESSOR_TYPE="abstractive"
export COMPRESSOR_MODEL="facebook/bart-large-cnn"
export LLM_MODEL="llama4:scout"
export NUM_SAMPLES=100

echo "Starting Llama 4 Scout with Abstractive Compression (BART) on all templates..."

for TEMPLATE in raw dev_mode aim dan; do
    export TEMPLATE
    echo "Running Llama with Abstractive - Template: $TEMPLATE"
    python3 pipeline/run_experiment.py --profile server --datasets jbb,advbench --rates 0.9,0.7,0.5,0.3,0.1 --delay 0.0
done

echo "ALL ABSTRACTIVE EXPERIMENTS DONE ON .238" > abstractive_done.flag
